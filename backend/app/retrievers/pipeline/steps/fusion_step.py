import re

from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.hybrid.fusion import rrf_fusion
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class FusionStep(BaseRetrieverStep):
    sparse_intent_patterns = (
        re.compile(r"第[0-9一二三四五六七八九十百千万几]+[章节条]"),
        re.compile(r"(章节|条款|法律|法规|劳动法)"),
    )

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        retrieval_intent = self._detect_retrieval_intent(context.active_query)
        sparse_boosted = retrieval_intent == "sparse"

        if sparse_boosted:
            context.fused_chunks = self._sparse_first_fusion(
                dense_chunks=context.dense_chunks,
                sparse_chunks=context.sparse_chunks,
                top_k=context.top_k,
            )
            fusion_strategy = "sparse_first"
        else:
            context.fused_chunks = rrf_fusion(
                dense_chunks=context.dense_chunks,
                sparse_chunks=context.sparse_chunks,
                top_k=context.top_k,
            )
            fusion_strategy = "rrf"

        context.metadata.update(
            {
                "retrieval_intent": retrieval_intent,
                "sparse_boosted": sparse_boosted,
                "fused_total": len(context.fused_chunks),
                "fusion": fusion_strategy,
            }
        )
        return context

    def _detect_retrieval_intent(self, query: str) -> str:
        normalized_query = query.strip()
        if any(pattern.search(normalized_query) for pattern in self.sparse_intent_patterns):
            return "sparse"
        return "hybrid"

    def _sparse_first_fusion(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        fused_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        dense_ranks = {chunk.id: rank for rank, chunk in enumerate(dense_chunks, start=1)}

        for sparse_rank, chunk in enumerate(sparse_chunks, start=1):
            if chunk.id in seen_ids:
                continue
            fusion_score = 1.0 + 1 / sparse_rank
            metadata = {
                **chunk.metadata,
                "fusion_strategy": "sparse_first",
                "sparse_boosted": True,
                "dense_rank": dense_ranks.get(chunk.id),
                "sparse_rank": sparse_rank,
                "fusion_score": fusion_score,
                "sparse_score": chunk.metadata.get("sparse_score", chunk.score),
            }
            fused_chunks.append(
                chunk.model_copy(update={"score": fusion_score, "metadata": metadata})
            )
            seen_ids.add(chunk.id)
            if len(fused_chunks) >= top_k:
                return fused_chunks

        for dense_rank, chunk in enumerate(dense_chunks, start=1):
            if chunk.id in seen_ids:
                continue
            fusion_score = 1 / (100 + dense_rank)
            metadata = {
                **chunk.metadata,
                "fusion_strategy": "sparse_first_dense_backfill",
                "sparse_boosted": True,
                "dense_rank": dense_rank,
                "sparse_rank": None,
                "fusion_score": fusion_score,
            }
            fused_chunks.append(
                chunk.model_copy(update={"score": fusion_score, "metadata": metadata})
            )
            seen_ids.add(chunk.id)
            if len(fused_chunks) >= top_k:
                break

        return fused_chunks
