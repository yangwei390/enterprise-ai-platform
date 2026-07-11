from backend.app.config.settings import settings
from backend.app.documents import StructureQueryHintParser
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
        context.fused_chunks, structure_boost_applied, hint_metadata = (
            self._apply_structure_boost(
                query=context.active_query,
                chunks=context.fused_chunks,
                top_k=context.top_k,
            )
        )
        context.metadata["soft_boost_applied"] = soft_boost_applied
        context.metadata["soft_boost_factor"] = self.soft_boost_factor
        context.metadata["structure_query_hint"] = hint_metadata
        context.metadata["structure_soft_boost_applied"] = structure_boost_applied
        context.metadata["structure_soft_boost_factor"] = settings.STRUCTURE_SOFT_BOOST_FACTOR
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

    def _apply_structure_boost(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], bool, dict]:
        if not settings.STRUCTURE_QUERY_HINT_ENABLED:
            return chunks, False, {"enabled": False}
        hint = StructureQueryHintParser().parse(query)
        metadata = hint.to_metadata()
        metadata["enabled"] = True
        if not hint.has_hint:
            return chunks, False, metadata

        boosted_chunks: list[RetrievedChunk] = []
        applied = False
        for chunk in chunks:
            if not self._matches_hint(chunk.metadata, metadata):
                boosted_chunks.append(chunk)
                continue
            boosted_score = chunk.score * settings.STRUCTURE_SOFT_BOOST_FACTOR
            boosted_chunks.append(
                chunk.model_copy(
                    update={
                        "score": boosted_score,
                        "metadata": {
                            **chunk.metadata,
                            "structure_soft_boost_applied": True,
                            "structure_soft_boost_factor": settings.STRUCTURE_SOFT_BOOST_FACTOR,
                            "score_before_structure_soft_boost": chunk.score,
                        },
                    }
                )
            )
            applied = True
        boosted_chunks.sort(key=lambda item: item.score, reverse=True)
        return boosted_chunks[:top_k], applied, metadata

    def _matches_hint(self, chunk_metadata: dict, hint_metadata: dict) -> bool:
        chapter_number = hint_metadata.get("chapter_number")
        article_number = hint_metadata.get("article_number")
        if chapter_number is not None and chunk_metadata.get("chapter_number") == chapter_number:
            return True
        if article_number is not None:
            start = chunk_metadata.get("article_start")
            end = chunk_metadata.get("article_end")
            if isinstance(start, int) and isinstance(end, int) and start <= article_number <= end:
                return True
            if chunk_metadata.get("article_number") == article_number:
                return True
        return False
