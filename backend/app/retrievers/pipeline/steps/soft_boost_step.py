from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class SoftBoostStep(BaseRetrieverStep):
    soft_boost_factor = 1.2

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        candidate_document_ids = (
            context.auto_filter_result.candidate_document_ids
            if context.auto_filter_result is not None
            else []
        )
        context.fused_chunks, soft_boost_applied = self._apply_soft_boost(
            chunks=context.fused_chunks,
            candidate_document_ids=candidate_document_ids,
            top_k=context.top_k,
        )
        context.metadata["soft_boost_applied"] = soft_boost_applied
        context.metadata["soft_boost_factor"] = self.soft_boost_factor
        return context

    def _apply_soft_boost(
        self,
        chunks: list[RetrievedChunk],
        candidate_document_ids: list[int],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], bool]:
        if not candidate_document_ids:
            return chunks, False

        candidate_ids = set(candidate_document_ids)
        boosted_chunks: list[RetrievedChunk] = []
        soft_boost_applied = False
        for chunk in chunks:
            if chunk.document_id not in candidate_ids:
                boosted_chunks.append(chunk)
                continue

            boosted_score = chunk.score * self.soft_boost_factor
            metadata = {
                **chunk.metadata,
                "auto_filter_candidate": True,
                "soft_boost_applied": True,
                "soft_boost_factor": self.soft_boost_factor,
                "score_before_soft_boost": chunk.score,
                "fusion_score": boosted_score,
            }
            boosted_chunks.append(
                chunk.model_copy(update={"score": boosted_score, "metadata": metadata})
            )
            soft_boost_applied = True

        boosted_chunks.sort(key=lambda item: item.score, reverse=True)
        return boosted_chunks[:top_k], soft_boost_applied
