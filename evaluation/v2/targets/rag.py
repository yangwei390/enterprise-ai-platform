import asyncio
from time import perf_counter
from typing import Any

from backend.app.rag import RagChatInput, RagChatPipeline
from backend.app.retrievers.pipeline import RetrieverPipeline, RetrieverPipelineContext

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.base import BaseEvaluationTarget, elapsed_ms


class RAGEvaluationTarget(BaseEvaluationTarget):
    name = "rag"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        started_at = perf_counter()
        mode = case.input.get("mode", "rag-answer")
        try:
            if mode == "retrieval-only":
                pipeline_context = await asyncio.to_thread(
                    RetrieverPipeline().run,
                    RetrieverPipelineContext(
                        query=case.query or "",
                        knowledge_base_id=case.input.get("knowledge_base_id"),
                        top_k=int(case.input.get("top_k", 5)),
                        score_threshold=case.input.get("score_threshold"),
                        metadata_filter=case.input.get("metadata_filter"),
                    ),
                )
                chunks = pipeline_context.context_chunks or pipeline_context.reranked_chunks
                return EvaluationTargetResult(
                    target=self.name,
                    input=case.input,
                    output={"context_text": pipeline_context.context_text},
                    chunks=[_chunk_to_dict(chunk) for chunk in chunks],
                    sources=[_chunk_to_dict(chunk) for chunk in chunks],
                    metadata=pipeline_context.metadata,
                    duration_ms=elapsed_ms(started_at),
                )

            result = await asyncio.to_thread(
                RagChatPipeline().run,
                RagChatInput(
                    query=case.query or "",
                    knowledge_base_id=case.input.get("knowledge_base_id"),
                    top_k=int(case.input.get("top_k", 5)),
                    score_threshold=case.input.get("score_threshold"),
                    metadata_filter=case.input.get("metadata_filter"),
                    model=case.input.get("model"),
                ),
            )
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                output={
                    "context_text": result.context_text,
                    "prompt_text": result.prompt_text,
                    "llm_model": result.llm_model,
                },
                answer=result.answer,
                sources=[_dump(item) for item in result.sources],
                citations=[_dump(item) for item in result.citations],
                chunks=[_dump(item) for item in result.sources],
                metadata=result.metadata,
                usage=result.metadata.get("llm_usage", {}),
                duration_ms=elapsed_ms(started_at),
            )
        except Exception as exc:
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                duration_ms=elapsed_ms(started_at),
                error=str(exc),
            )


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "id": getattr(chunk, "id", None),
        "document_id": getattr(chunk, "document_id", None),
        "knowledge_base_id": getattr(chunk, "knowledge_base_id", None),
        "chunk_index": getattr(chunk, "chunk_index", None),
        "source": getattr(chunk, "source", None),
        "score": getattr(chunk, "score", None),
        "text": getattr(chunk, "text", None),
        "metadata": getattr(chunk, "metadata", {}) or {},
    }


def _dump(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"value": item}
