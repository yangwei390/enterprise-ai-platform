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


def _to_trace_chunk(chunk: Any, text_limit: int = 300) -> RagTraceChunk:
    metadata = getattr(chunk, "metadata", {}) or {}
    source = getattr(chunk, "source", None) or _metadata_value(metadata, "source")
    text = getattr(chunk, "text", "") or ""
    score = getattr(chunk, "score", None)
    rerank_score = getattr(chunk, "rerank_score", None)
    original_score = getattr(chunk, "original_score", None)

    return RagTraceChunk(
        id=getattr(chunk, "id", None),
        document_id=getattr(chunk, "document_id", None),
        knowledge_base_id=getattr(chunk, "knowledge_base_id", None),
        chunk_index=getattr(chunk, "chunk_index", None),
        source=source,
        text_preview=text[:text_limit],
        score=score if score is not None else original_score,
        dense_rank=_metadata_value(metadata, "dense_rank"),
        sparse_rank=_metadata_value(metadata, "sparse_rank"),
        fusion_score=_metadata_value(metadata, "fusion_score"),
        rerank_score=rerank_score,
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

    trace_result = RagTraceResult(
        query=request.query,
        rewritten_query=rewritten_query,
        knowledge_base_id=request.knowledge_base_id,
        retriever_mode=retrieve_result.metadata.get("retriever_mode", "hybrid"),
        dense_chunks=[],
        sparse_chunks=[],
        fused_chunks=[_to_trace_chunk(chunk) for chunk in retrieve_result.chunks],
        reranked_chunks=[_to_trace_chunk(chunk) for chunk in rerank_result.chunks],
        context_chunks=[_to_trace_chunk(chunk) for chunk in compression_result.chunks],
        context_text_preview=compression_result.context_text[:1000],
        metadata={
            "trace_limited": True,
            "trace_limited_reason": "HybridRetriever currently exposes fused chunks only.",
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
    return success(data=trace_result)
