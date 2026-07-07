from typing import Any

from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.context.compression.compressor import SimpleContextCompressor
from backend.app.debug import RagTraceChunk, RagTraceResult
from backend.app.query.rewriter import SimpleQueryRewriter
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RagTraceRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 10
    score_threshold: float | None = 0.0
    metadata_filter: dict | None = None


def _metadata_value(metadata: dict, key: str) -> Any | None:
    value = metadata.get(key)
    if value is not None:
        return value
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        return nested_metadata.get(key)
    return None


def _optional_int(value: Any | None) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any | None) -> float | None:
    return value if isinstance(value, int | float) else None


def _first_float(*values: Any | None) -> float | None:
    for value in values:
        float_value = _optional_float(value)
        if float_value is not None:
            return float_value
    return None


def _to_trace_chunk(chunk: Any, text_limit: int = 300) -> RagTraceChunk:
    metadata = getattr(chunk, "metadata", {}) or {}
    source = getattr(chunk, "source", None) or _metadata_value(metadata, "source")
    text = getattr(chunk, "text", "") or ""
    score = getattr(chunk, "score", None)
    rerank_score = getattr(chunk, "rerank_score", None)
    original_score = getattr(chunk, "original_score", None)
    fusion_score = _optional_float(_metadata_value(metadata, "fusion_score"))

    return RagTraceChunk(
        id=getattr(chunk, "id", None),
        document_id=getattr(chunk, "document_id", None),
        knowledge_base_id=getattr(chunk, "knowledge_base_id", None),
        chunk_index=getattr(chunk, "chunk_index", None),
        source=source,
        text_preview=text[:text_limit],
        score=_first_float(score, original_score, fusion_score, rerank_score),
        dense_rank=_optional_int(_metadata_value(metadata, "dense_rank")),
        sparse_rank=_optional_int(_metadata_value(metadata, "sparse_rank")),
        fusion_score=fusion_score,
        rerank_score=_optional_float(rerank_score),
        metadata=metadata,
    )


@router.post("/debug/rag-trace", response_model=ApiResponse)
def rag_trace(request: RagTraceRequest) -> ApiResponse:
    rewrite_result = SimpleQueryRewriter().rewrite(request.query)
    rewritten_query = rewrite_result.rewritten_query

    retriever = RetrieverFactory.get_hybrid_retriever()
    retrieve_result = retriever.retrieve(
        HybridRetrieveQuery(
            query=rewritten_query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            metadata_filter=request.metadata_filter,
        )
    )

    reranker = RerankerFactory.get_reranker()
    rerank_result = reranker.rerank(
        RerankQuery(
            query=rewritten_query,
            chunks=retrieve_result.chunks,
            top_k=request.top_k,
        )
    )

    context_builder = ContextBuilderFactory.get_builder()
    context_result = context_builder.build(
        ContextBuildRequest(
            query=rewritten_query,
            chunks=rerank_result.chunks,
        )
    )

    compression_result = SimpleContextCompressor().compress(
        context_text=context_result.context_text,
        chunks=context_result.chunks,
    )

    fused_chunks = retrieve_result.chunks[: request.top_k]
    context_chunks = compression_result.chunks

    trace_result = RagTraceResult(
        query=request.query,
        rewritten_query=rewritten_query,
        knowledge_base_id=request.knowledge_base_id,
        retriever_mode=retrieve_result.metadata.get("retriever_mode", "hybrid"),
        dense_chunks=[],
        sparse_chunks=[],
        fused_chunks=[_to_trace_chunk(chunk) for chunk in fused_chunks],
        reranked_chunks=[_to_trace_chunk(chunk) for chunk in rerank_result.chunks],
        context_chunks=[_to_trace_chunk(chunk) for chunk in context_chunks],
        context_text_preview=compression_result.context_text[:1000],
        metadata={
            "trace_limited": True,
            "trace_limited_reason": "HybridRetriever currently exposes fused chunks only.",
            "fused_chunk_count": len(fused_chunks),
            "context_chunk_count": len(context_chunks),
            "context_text_preview_chars": len(compression_result.context_text[:1000]),
            "query_rewrite": rewrite_result.model_dump(),
            "retriever": retrieve_result.metadata,
            "reranker": rerank_result.metadata,
            "context_builder": context_result.metadata,
            "context_compression": {
                "original_chars": compression_result.original_chars,
                "compressed_chars": compression_result.compressed_chars,
                "compression_applied": compression_result.compression_applied,
                **compression_result.metadata,
            },
        },
    )
    return success(data=trace_result.model_dump())
