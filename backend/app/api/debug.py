from pathlib import Path
from typing import Any, cast

from backend.app.api.evaluation import evaluation_debug_state
from backend.app.chunkers import ChunkerFactory
from backend.app.chunkers.router import ChunkStrategyRouter
from backend.app.cleaners import CleanerFactory
from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.context_compression import (
    CompressionInput,
    CompressionResult,
    ContextCompressorFactory,
    get_context_compression_config,
)
from backend.app.db.session import get_db
from backend.app.debug import RagTraceChunk, RagTraceResult
from backend.app.documents import DocumentClassifier, StructureParserFactory
from backend.app.logger import logger
from backend.app.mcp.client_manager import get_mcp_client_manager
from backend.app.memory.factory import MemoryFactory
from backend.app.parsers import ParserFactory
from backend.app.query.rewriter import SimpleQueryRewriter
from backend.app.repositories.document import DocumentRepository
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.hybrid.sparse_retriever import BM25SparseRetriever
from backend.app.retrievers.pipeline import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps import MMRStep, NeighborExpansionStep
from backend.app.schemas import ApiResponse, success
from backend.app.tools import get_tool_registry
from backend.app.workflows.factory import WorkflowRuntimeFactory
from backend.app.workflows.langgraph import list_workflow_definitions_v2
from backend.app.workflows.langgraph.runtime import LangGraphWorkflowRuntime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter()


class RagTraceRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 10
    score_threshold: float | None = 0.0
    metadata_filter: dict | None = None


class ChunkPreviewRequest(BaseModel):
    strategy: str | None = None


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


def _to_trace_chunk(
    chunk: Any,
    text_limit: int = 300,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RagTraceChunk:
    metadata = getattr(chunk, "metadata", {}) or {}
    source = getattr(chunk, "source", None) or _metadata_value(metadata, "source")
    text = getattr(chunk, "text", "") or ""
    score = getattr(chunk, "score", None)
    rerank_score = getattr(chunk, "rerank_score", None)
    original_score = getattr(chunk, "original_score", None)
    fusion_score = _optional_float(_metadata_value(metadata, "fusion_score"))
    sparse_score = _optional_float(_metadata_value(metadata, "sparse_score"))

    return RagTraceChunk(
        id=getattr(chunk, "id", None),
        document_id=getattr(chunk, "document_id", None),
        knowledge_base_id=getattr(chunk, "knowledge_base_id", None),
        chunk_index=getattr(chunk, "chunk_index", None),
        source=source,
        text_preview=text[:text_limit],
        score=_first_float(score, original_score, fusion_score, rerank_score),
        dense_rank=dense_rank or _optional_int(_metadata_value(metadata, "dense_rank")),
        sparse_rank=sparse_rank or _optional_int(_metadata_value(metadata, "sparse_rank")),
        fusion_score=fusion_score,
        sparse_score=sparse_score,
        rerank_score=_optional_float(rerank_score),
        metadata=metadata,
    )


def _compress_context(
    query: str,
    context_text: str,
    chunks: list[Any],
) -> tuple[str, list[Any], CompressionResult, dict]:
    config = get_context_compression_config()
    base_metadata = {
        "enabled": config.enabled,
        "provider": config.provider,
        "original_chunk_count": len(chunks),
        "compressed_chunk_count": len(chunks),
        "original_chars": len(context_text),
        "compressed_chars": len(context_text),
        "skipped_chunk_count": 0,
        "max_chars": config.max_chars,
        "max_chunk_chars": config.max_chunk_chars,
        "failed": False,
        "error": None,
    }

    if not config.enabled:
        result = CompressionResult(
            compressed_chunks=chunks,
            original_chunk_count=len(chunks),
            compressed_chunk_count=len(chunks),
            original_chars=len(context_text),
            compressed_chars=len(context_text),
            skipped_chunk_count=0,
            metadata={"context_text": context_text},
        )
        return context_text, chunks, result, base_metadata

    try:
        result = ContextCompressorFactory.get_compressor(config.provider).compress(
            CompressionInput(
                query=query,
                chunks=chunks,
                max_chars=config.max_chars,
                metadata={"context_text": context_text},
            )
        )
        compressed_context_text = str(result.metadata.get("context_text") or "")
        metadata = {
            **base_metadata,
            "original_chunk_count": result.original_chunk_count,
            "compressed_chunk_count": result.compressed_chunk_count,
            "original_chars": result.original_chars,
            "compressed_chars": result.compressed_chars,
            "skipped_chunk_count": result.skipped_chunk_count,
            **result.metadata,
            "failed": False,
            "error": None,
        }
        return compressed_context_text, result.compressed_chunks, result, metadata
    except Exception as exc:
        logger.exception("Context compression failed in debug API")
        if not config.fail_open:
            raise
        result = CompressionResult(
            compressed_chunks=chunks,
            original_chunk_count=len(chunks),
            compressed_chunk_count=len(chunks),
            original_chars=len(context_text),
            compressed_chars=len(context_text),
            skipped_chunk_count=0,
            metadata={"context_text": context_text},
        )
        metadata = {
            **base_metadata,
            "failed": True,
            "error": str(exc),
        }
        return context_text, chunks, result, metadata


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
    mmr_context = MMRStep().run(
        RetrieverPipelineContext(
            query=rewritten_query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            reranked_chunks=rerank_result.chunks,
        )
    )
    neighbor_context = NeighborExpansionStep().run(
        RetrieverPipelineContext(
            query=rewritten_query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            reranked_chunks=mmr_context.reranked_chunks,
        )
    )

    context_builder = ContextBuilderFactory.get_builder()
    context_result = context_builder.build(
        ContextBuildRequest(
            query=rewritten_query,
            chunks=neighbor_context.reranked_chunks,
        )
    )

    (
        compressed_context_text,
        compressed_chunks,
        _compression_result,
        compression_metadata,
    ) = _compress_context(
        query=rewritten_query,
        context_text=context_result.context_text,
        chunks=context_result.chunks,
    )

    fused_chunks = retrieve_result.chunks[: request.top_k]
    context_chunks = compressed_chunks

    trace_result = RagTraceResult(
        query=request.query,
        rewritten_query=rewritten_query,
        knowledge_base_id=request.knowledge_base_id,
        retriever_mode=retrieve_result.metadata.get("retriever_mode", "hybrid"),
        dense_chunks=[],
        sparse_chunks=[],
        fused_chunks=[_to_trace_chunk(chunk) for chunk in fused_chunks],
        reranked_chunks=[
            _to_trace_chunk(chunk) for chunk in neighbor_context.reranked_chunks
        ],
        context_chunks=[_to_trace_chunk(chunk) for chunk in context_chunks],
        context_text_preview=compressed_context_text[:1000],
        metadata={
            "trace_limited": True,
            "trace_limited_reason": "HybridRetriever currently exposes fused chunks only.",
            "fused_chunk_count": len(fused_chunks),
            "context_chunk_count": len(context_chunks),
            "context_text_preview_chars": len(compressed_context_text[:1000]),
            "query_rewrite": rewrite_result.model_dump(),
            "retriever": retrieve_result.metadata,
            "reranker": rerank_result.metadata,
            "mmr": mmr_context.metadata.get("mmr", {}),
            "neighbor_expansion": neighbor_context.metadata.get(
                "neighbor_expansion", {}
            ),
            "context_builder": context_result.metadata,
            "context_compression": compression_metadata,
        },
    )
    return success(data=trace_result.model_dump())


@router.post("/debug/retriever-compare", response_model=ApiResponse)
def retriever_compare(request: RagTraceRequest) -> ApiResponse:
    rewrite_result = SimpleQueryRewriter().rewrite(request.query)
    rewritten_query = rewrite_result.rewritten_query
    query = HybridRetrieveQuery(
        query=rewritten_query,
        knowledge_base_id=request.knowledge_base_id,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        metadata_filter=request.metadata_filter,
    )
    metadata: dict[str, Any] = {
        "query_rewrite": rewrite_result.model_dump(),
        "dense": {},
        "sparse": {},
        "fusion": {},
        "context": {},
        "context_compression": {},
    }

    dense_chunks = []
    try:
        dense_chunks = DenseRetriever().retrieve(query)
        metadata["dense"] = {
            "available": True,
            "total": len(dense_chunks),
        }
    except Exception as exc:
        metadata["dense"] = {
            "available": False,
            "unavailable_reason": str(exc),
        }

    sparse_chunks = []
    try:
        sparse_chunks = BM25SparseRetriever().retrieve(query)
        metadata["sparse"] = {
            "available": True,
            "total": len(sparse_chunks),
        }
    except Exception as exc:
        metadata["sparse"] = {
            "available": False,
            "unavailable_reason": str(exc),
        }

    fused_chunks = []
    try:
        retrieve_result = RetrieverFactory.get_hybrid_retriever().retrieve(query)
        fused_chunks = retrieve_result.chunks[: request.top_k]
        metadata["fusion"] = {
            "available": True,
            **retrieve_result.metadata,
        }
    except Exception as exc:
        metadata["fusion"] = {
            "available": False,
            "unavailable_reason": str(exc),
        }

    context_chunks = []
    context_text_preview = ""
    reranked_chunks = []
    if fused_chunks:
        try:
            rerank_result = RerankerFactory.get_reranker().rerank(
                RerankQuery(
                    query=rewritten_query,
                    chunks=fused_chunks,
                    top_k=request.top_k,
                )
            )
            reranked_chunks = rerank_result.chunks
            mmr_context = MMRStep().run(
                RetrieverPipelineContext(
                    query=rewritten_query,
                    knowledge_base_id=request.knowledge_base_id,
                    top_k=request.top_k,
                    reranked_chunks=reranked_chunks,
                )
            )
            reranked_chunks = mmr_context.reranked_chunks
            neighbor_context = NeighborExpansionStep().run(
                RetrieverPipelineContext(
                    query=rewritten_query,
                    knowledge_base_id=request.knowledge_base_id,
                    top_k=request.top_k,
                    reranked_chunks=reranked_chunks,
                )
            )
            reranked_chunks = neighbor_context.reranked_chunks
            context_result = ContextBuilderFactory.get_builder().build(
                ContextBuildRequest(
                    query=rewritten_query,
                    chunks=reranked_chunks,
                )
            )
            (
                compressed_context_text,
                compressed_chunks,
                _compression_result,
                compression_metadata,
            ) = _compress_context(
                query=rewritten_query,
                context_text=context_result.context_text,
                chunks=context_result.chunks,
            )
            context_chunks = compressed_chunks
            context_text_preview = compressed_context_text[:1000]
            metadata["context"] = {
                "available": True,
                "reranker": rerank_result.metadata,
                "mmr": mmr_context.metadata.get("mmr", {}),
                "neighbor_expansion": neighbor_context.metadata.get(
                    "neighbor_expansion", {}
                ),
                "context_builder": context_result.metadata,
                "context_compression": compression_metadata,
            }
            metadata["context_compression"] = compression_metadata
        except Exception as exc:
            metadata["context"] = {
                "available": False,
                "unavailable_reason": str(exc),
            }
    else:
        metadata["context"] = {
            "available": False,
            "unavailable_reason": "No fused chunks available.",
        }

    result = RagTraceResult(
        query=request.query,
        rewritten_query=rewritten_query,
        knowledge_base_id=request.knowledge_base_id,
        retriever_mode="compare",
        dense_chunks=[
            _to_trace_chunk(chunk, text_limit=500, dense_rank=index)
            for index, chunk in enumerate(dense_chunks, start=1)
        ],
        sparse_chunks=[
            _to_trace_chunk(chunk, text_limit=500, sparse_rank=index)
            for index, chunk in enumerate(sparse_chunks, start=1)
        ],
        fused_chunks=[
            _to_trace_chunk(chunk, text_limit=500) for chunk in fused_chunks
        ],
        reranked_chunks=[
            _to_trace_chunk(chunk, text_limit=500) for chunk in reranked_chunks
        ],
        context_chunks=[
            _to_trace_chunk(chunk, text_limit=500) for chunk in context_chunks
        ],
        context_text_preview=context_text_preview,
        metadata=metadata,
    )
    return success(data=result.model_dump())


@router.get("/debug/documents/{document_id}/structure", response_model=ApiResponse)
def debug_document_structure(
    document_id: int,
    db: Session = Depends(get_db),
) -> ApiResponse:
    preview = _build_document_chunk_preview(document_id=document_id, db=db, strategy="auto")
    structure = preview["structure"]
    nodes = structure.get("nodes", [])
    summary_nodes = [
        {
            "id": node["id"],
            "node_type": node["node_type"],
            "title": node.get("title"),
            "level": node.get("level"),
            "path": node.get("path"),
            "metadata": node.get("metadata"),
        }
        for node in nodes
        if node["node_type"] in {"document", "chapter", "section", "heading"}
    ]
    return success(
        data={
            "document_id": document_id,
            "document_type": structure.get("document_type"),
            "node_count": structure.get("metadata", {}).get("node_count", len(nodes)),
            "max_depth": structure.get("metadata", {}).get("max_depth", 0),
            "nodes": summary_nodes[:200],
            "metadata": structure.get("metadata", {}),
        }
    )


@router.get("/debug/documents/{document_id}/chunks", response_model=ApiResponse)
def debug_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
) -> ApiResponse:
    preview = _build_document_chunk_preview(document_id=document_id, db=db, strategy="auto")
    return success(
        data={
            "document_id": document_id,
            "strategy": preview["chunk_result"]["strategy"],
            "total_chunks": preview["chunk_result"]["total_chunks"],
            "chunks": preview["chunks"],
            "metadata": preview["chunk_result"]["metadata"],
        }
    )


@router.post("/debug/documents/{document_id}/chunk-preview", response_model=ApiResponse)
def debug_document_chunk_preview(
    document_id: int,
    request: ChunkPreviewRequest,
    db: Session = Depends(get_db),
) -> ApiResponse:
    return success(
        data=_build_document_chunk_preview(
            document_id=document_id,
            db=db,
            strategy=request.strategy or "auto",
        )
    )


def _build_document_chunk_preview(document_id: int, db: Session, strategy: str) -> dict:
    document = DocumentRepository(db).get(document_id)
    if document is None:
        return {"found": False, "document_id": document_id}
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.is_absolute():
        upload_dir = PROJECT_ROOT / upload_dir
    file_path = upload_dir / str(document.storage_path)
    parser = ParserFactory.get_parser(file_path)
    parse_result = parser.parse(file_path)
    cleaner = CleanerFactory.get_cleaner(file_path.suffix.lower())
    clean_result = cleaner.clean(parse_result.text)
    base_metadata = {
        "document_id": document.id,
        "knowledge_base_id": document.knowledge_base_id,
        "source": document.original_filename or document.filename,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "mime_type": document.mime_type,
        "parser": parser.__class__.__name__,
        "cleaner": cleaner.__class__.__name__,
        "page_count": parse_result.page_count,
        "suffix": file_path.suffix.lower(),
    }
    classification = DocumentClassifier().classify(
        text=clean_result.text,
        filename=document.original_filename or document.filename,
        mime_type=document.mime_type,
        metadata=base_metadata,
    )
    structure = StructureParserFactory.parse(
        clean_result.text,
        {**base_metadata, "document_type": classification.document_type},
        classification.document_type,
    )
    decision = ChunkStrategyRouter().route(
        document_type=structure.document_type,
        structure=structure,
        metadata=base_metadata,
        requested_strategy=strategy,
    )
    chunk_metadata = {
        **base_metadata,
        "document_type": structure.document_type,
        "_document_structure": structure,
        "document_classification": classification.model_dump(),
        "document_structure": structure.metadata,
        **decision.to_metadata(),
    }
    chunker = ChunkerFactory.get_chunker(file_path.suffix.lower(), decision.actual_strategy)
    chunk_result = chunker.chunk(clean_result.text, chunk_metadata)
    for chunk in chunk_result.chunks:
        chunk.metadata.pop("_document_structure", None)
    return {
        "found": True,
        "document_id": document_id,
        "classification": classification.model_dump(),
        "structure": structure.model_dump(),
        "chunk_result": {
            **chunk_result.model_dump(exclude={"chunks"}),
            "metadata": {
                key: value
                for key, value in chunk_result.metadata.items()
                if key != "_document_structure"
            },
        },
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "chunk_uid": chunk.metadata.get("chunk_uid"),
                "text_preview": chunk.text[:300],
                "chunk_role": chunk.metadata.get("chunk_role"),
                "parent_chunk_id": chunk.metadata.get("parent_chunk_id"),
                "section_path": chunk.metadata.get("section_path"),
                "metadata": chunk.metadata,
            }
            for chunk in chunk_result.chunks[:200]
        ],
    }


@router.get("/debug/tools", response_model=ApiResponse)
def debug_tools() -> ApiResponse:
    return success(data=get_tool_registry().snapshot())


@router.post("/debug/tools/refresh", response_model=ApiResponse)
def refresh_debug_tools() -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(
            data={
                "refreshed": False,
                "reason": "tool registry refresh is disabled in production",
            }
        )
    result = get_tool_registry().refresh()
    return success(data=result)


@router.get("/debug/memory", response_model=ApiResponse)
def debug_memory() -> ApiResponse:
    manager = MemoryFactory.get_manager()
    return success(data=manager.snapshot().model_dump())


@router.get("/debug/cache", response_model=ApiResponse)
def debug_cache() -> ApiResponse:
    snapshot = MemoryFactory.get_manager().snapshot()
    return success(
        data={
            "provider": snapshot.provider,
            "cache_count": snapshot.cache_count,
            "metadata": snapshot.metadata,
        }
    )


@router.delete("/debug/cache", response_model=ApiResponse)
def delete_debug_cache(cache_key: str | None = None) -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"deleted": False, "reason": "disabled in production"})
    MemoryFactory.get_manager().delete_cache(cache_key)
    return success(data={"deleted": True, "cache_key": cache_key})


@router.delete("/debug/memory/session/{session_id}", response_model=ApiResponse)
def delete_debug_session(session_id: str) -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"deleted": False, "reason": "disabled in production"})
    MemoryFactory.get_manager().delete_session(session_id)
    return success(data={"deleted": True, "session_id": session_id})


@router.get("/debug/checkpoints", response_model=ApiResponse)
def debug_checkpoints() -> ApiResponse:
    manager = MemoryFactory.get_checkpoint_manager()
    return success(data={"checkpoints": manager.list()})


@router.get("/debug/workflows", response_model=ApiResponse)
def debug_workflows() -> ApiResponse:
    definitions = list_workflow_definitions_v2()
    return success(
        data={
            "runtime": settings.WORKFLOW_RUNTIME,
            "workflows": [
                {
                    "id": definition.id,
                    "name": definition.name,
                    "version": definition.version,
                    "node_count": len(definition.nodes),
                    "edge_count": len(definition.edges),
                    "checkpoint_enabled": definition.checkpoint_enabled,
                    "approval_enabled": definition.approval_enabled,
                }
                for definition in definitions
            ],
        }
    )


@router.get("/debug/workflows/{workflow_id}", response_model=ApiResponse)
def debug_workflow_definition(workflow_id: str) -> ApiResponse:
    for definition in list_workflow_definitions_v2():
        if definition.id == workflow_id:
            return success(data=definition.model_dump())
    return success(data={"found": False, "workflow_id": workflow_id})


@router.get("/debug/workflows/runs/{thread_id}", response_model=ApiResponse)
def debug_workflow_run(thread_id: str) -> ApiResponse:
    runtime = cast(LangGraphWorkflowRuntime, WorkflowRuntimeFactory.get_runtime("langgraph"))
    state = runtime.get_state(thread_id)
    return success(data={"thread_id": thread_id, "state": state})


@router.get("/debug/workflows/checkpoints/{thread_id}", response_model=ApiResponse)
def debug_workflow_checkpoint(thread_id: str) -> ApiResponse:
    manager = MemoryFactory.get_checkpoint_manager()
    return success(
        data={
            "thread_id": thread_id,
            "checkpoint": manager.load(f"workflow:{thread_id}"),
        }
    )


@router.delete("/debug/workflows/checkpoints/{thread_id}", response_model=ApiResponse)
def delete_debug_workflow_checkpoint(thread_id: str) -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"deleted": False, "reason": "disabled in production"})
    MemoryFactory.get_checkpoint_manager().delete(f"workflow:{thread_id}")
    return success(data={"deleted": True, "thread_id": thread_id})


@router.get("/debug/mcp", response_model=ApiResponse)
async def debug_mcp() -> ApiResponse:
    manager = get_mcp_client_manager()
    health = await manager.health_all()
    registry_snapshot = get_tool_registry().snapshot()
    mcp_tools = [
        tool
        for tool in registry_snapshot.get("tools", [])
        if tool.get("provider") == "mcp"
    ]
    return success(
        data={
            "enabled": settings.MCP_ENABLED or settings.MCP_TOOL_PROVIDER_ENABLED,
            "configured_servers": manager.list_servers(),
            "health": health,
            "discovered_tool_count": len(mcp_tools),
            "tools": mcp_tools,
            "registry_version": registry_snapshot.get("registry_version"),
            "audit": manager.audit_records[-50:],
        }
    )


@router.post("/debug/mcp/connect/{server_name}", response_model=ApiResponse)
async def debug_mcp_connect(server_name: str) -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"connected": False, "reason": "disabled in production"})
    await get_mcp_client_manager().connect(server_name)
    return success(data={"connected": True, "server_name": server_name})


@router.post("/debug/mcp/disconnect/{server_name}", response_model=ApiResponse)
async def debug_mcp_disconnect(server_name: str) -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"disconnected": False, "reason": "disabled in production"})
    await get_mcp_client_manager().disconnect(server_name)
    return success(data={"disconnected": True, "server_name": server_name})


@router.post("/debug/mcp/refresh", response_model=ApiResponse)
async def debug_mcp_refresh() -> ApiResponse:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return success(data={"refreshed": False, "reason": "disabled in production"})
    result = await get_tool_registry().arefresh()
    return success(data=result)


@router.get("/debug/mcp/health", response_model=ApiResponse)
async def debug_mcp_health() -> ApiResponse:
    return success(data=await get_mcp_client_manager().health_all())


@router.get("/debug/evaluation", response_model=ApiResponse)
def debug_evaluation() -> ApiResponse:
    return success(data=evaluation_debug_state())
