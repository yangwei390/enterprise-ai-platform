import time

from backend.app.rerankers import (
    RerankerFactory,
    RerankInputItem,
    RerankResultItem,
)
from backend.app.rerankers.config import get_reranker_config
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class RerankStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        config = get_reranker_config()
        candidates = context.fused_chunks[: config.top_k]
        remaining_chunks = context.fused_chunks[config.top_k :]
        metadata = {
            "rerank_enabled": True,
            "rerank_provider": config.provider,
            "rerank_model": config.model,
            "rerank_top_k": config.top_k,
            "rerank_before_count": len(context.fused_chunks),
            "rerank_after_count": 0,
            "rerank_time_ms": 0.0,
            "rerank_failed": False,
            "rerank_error": None,
        }

        started_at = time.perf_counter()
        try:
            result_items, actual_provider, actual_model = self._rerank_candidates(
                query=context.active_query,
                chunks=candidates,
                top_k=min(context.top_k, len(candidates)),
                provider=config.provider,
            )
        except Exception as exc:
            metadata["rerank_failed"] = True
            metadata["rerank_error"] = str(exc)
            context.add_error("RerankStep", exc)
            if not config.fail_open:
                context.metadata["reranker"] = metadata
                raise
            result_items, actual_provider, actual_model = self._rerank_candidates(
                query=context.active_query,
                chunks=candidates,
                top_k=min(context.top_k, len(candidates)),
                provider="dummy",
            )
            metadata["rerank_fail_open"] = True
        metadata["rerank_provider"] = actual_provider
        metadata["rerank_model"] = actual_model

        metadata["rerank_time_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        context.reranked_chunks = [
            self._to_reranked_chunk(
                chunk=candidates[item.index],
                item=item,
                rank=rank,
                provider=str(metadata["rerank_provider"]),
                model=str(metadata["rerank_model"]),
            )
            for rank, item in enumerate(result_items, start=1)
            if item.index < len(candidates)
        ]

        used_ids = {chunk.id for chunk in context.reranked_chunks}
        for chunk in [*candidates, *remaining_chunks]:
            if len(context.reranked_chunks) >= context.top_k:
                break
            if chunk.id in used_ids:
                continue
            context.reranked_chunks.append(
                self._to_reranked_chunk(
                    chunk=chunk,
                    item=RerankResultItem(
                        id=chunk.id,
                        index=0,
                        score=chunk.score,
                        metadata={
                            "provider": "fallback",
                            "model": "original_score",
                        },
                    ),
                    rank=len(context.reranked_chunks) + 1,
                    provider="fallback",
                    model="original_score",
                )
            )

        metadata["rerank_after_count"] = len(context.reranked_chunks)
        context.metadata["reranker"] = metadata
        context.metadata["reranked_total"] = len(context.reranked_chunks)
        return context

    def _rerank_candidates(
        self,
        query: str,
        chunks,
        top_k: int,
        provider: str | None = None,
    ) -> tuple[list[RerankResultItem], str, str]:
        input_items = [
            RerankInputItem(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                original_score=chunk.score,
            )
            for chunk in chunks
        ]
        reranker = RerankerFactory.get_reranker(provider=provider)
        result = reranker.rerank(
            query=query,
            items=input_items,
            top_k=top_k,
        )
        result_items = result if isinstance(result, list) else []
        return result_items, reranker.provider, reranker.model_name

    def _to_reranked_chunk(
        self,
        chunk,
        item: RerankResultItem,
        rank: int,
        provider: str,
        model: str,
    ):
        from backend.app.rerankers import RerankedChunk

        metadata = {
            **chunk.metadata,
            **item.metadata,
            "rerank_score": item.score,
            "rerank_rank": rank,
            "rerank_provider": provider,
            "rerank_model": model,
        }
        return RerankedChunk(
            id=chunk.id,
            original_score=chunk.score,
            rerank_score=item.score,
            text=chunk.text,
            document_id=chunk.document_id,
            knowledge_base_id=chunk.knowledge_base_id,
            chunk_index=chunk.chunk_index,
            metadata=metadata,
        )
