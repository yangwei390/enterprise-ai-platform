import re
from dataclasses import dataclass

from backend.app.config.settings import settings
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


@dataclass
class MMRConfig:
    enabled: bool = True
    lambda_value: float = 0.7
    top_k: int = 5
    min_score: float = 0.0
    fail_open: bool = True
    similarity_threshold: float = 0.85


def get_mmr_config() -> MMRConfig:
    return MMRConfig(
        enabled=settings.MMR_ENABLED,
        lambda_value=settings.MMR_LAMBDA,
        top_k=settings.MMR_TOP_K,
        min_score=settings.MMR_MIN_SCORE,
        fail_open=settings.MMR_FAIL_OPEN,
        similarity_threshold=settings.MMR_SIMILARITY_THRESHOLD,
    )


def text_similarity(left: str, right: str) -> float:
    left_tokens = _text_ngrams(left)
    right_tokens = _text_ngrams(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _text_ngrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


class MMRStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        config = get_mmr_config()
        metadata = self._base_metadata(config, len(context.reranked_chunks))
        if not config.enabled:
            context.metadata["mmr"] = metadata
            return context

        original_chunks = context.reranked_chunks
        try:
            selected_chunks = self._select_chunks(original_chunks, config)
            context.reranked_chunks = selected_chunks
            context.metadata["mmr"] = {
                **metadata,
                "selected_chunk_count": len(selected_chunks),
                "removed_chunk_count": len(original_chunks) - len(selected_chunks),
                "failed": False,
                "error": None,
            }
            return context
        except Exception as exc:
            context.add_error("MMRStep", exc)
            metadata["failed"] = True
            metadata["error"] = str(exc)
            context.metadata["mmr"] = metadata
            if not config.fail_open:
                raise RuntimeError(f"MMR failed: {exc}") from exc
            context.reranked_chunks = original_chunks
            return context

    def _select_chunks(
        self,
        chunks: list[RerankedChunk],
        config: MMRConfig,
    ) -> list[RerankedChunk]:
        protected_chunks = [
            chunk for chunk in chunks if chunk.metadata.get("neighbor_expanded") is True
        ]
        candidates = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("neighbor_expanded") is not True
            and self._relevance_score(chunk) >= config.min_score
        ]
        if not candidates:
            return chunks

        selected: list[RerankedChunk] = []
        remaining = list(candidates)
        while remaining and len(selected) < config.top_k:
            best_chunk: RerankedChunk | None = None
            best_mmr_score: float | None = None
            best_similarity = 0.0

            for candidate in remaining:
                max_similarity = self._max_similarity(candidate, selected)
                if selected and max_similarity >= config.similarity_threshold:
                    continue

                relevance_score = self._relevance_score(candidate)
                mmr_score = (
                    config.lambda_value * relevance_score
                    - (1 - config.lambda_value) * max_similarity
                )
                if best_mmr_score is None or mmr_score > best_mmr_score:
                    best_chunk = candidate
                    best_mmr_score = mmr_score
                    best_similarity = max_similarity

            if best_chunk is None or best_mmr_score is None:
                break

            remaining.remove(best_chunk)
            selected.append(
                self._mark_selected(
                    chunk=best_chunk,
                    rank=len(selected) + 1,
                    mmr_score=best_mmr_score,
                    max_similarity=best_similarity,
                    config=config,
                )
            )

        return [*selected, *protected_chunks]

    def _mark_selected(
        self,
        *,
        chunk: RerankedChunk,
        rank: int,
        mmr_score: float,
        max_similarity: float,
        config: MMRConfig,
    ) -> RerankedChunk:
        metadata = {
            **chunk.metadata,
            "mmr_selected": True,
            "mmr_rank": rank,
            "mmr_score": mmr_score,
            "mmr_relevance_score": self._relevance_score(chunk),
            "mmr_max_similarity": max_similarity,
            "mmr_lambda": config.lambda_value,
        }
        return chunk.model_copy(update={"metadata": metadata})

    def _max_similarity(
        self,
        candidate: RerankedChunk,
        selected: list[RerankedChunk],
    ) -> float:
        if not selected:
            return 0.0
        return max(
            text_similarity(candidate.text, selected_chunk.text)
            for selected_chunk in selected
        )

    def _relevance_score(self, chunk: RerankedChunk) -> float:
        for value in (
            chunk.metadata.get("rerank_score"),
            chunk.rerank_score,
            chunk.metadata.get("score"),
            getattr(chunk, "score", None),
        ):
            if isinstance(value, int | float):
                return float(value)
        return 0.0

    def _base_metadata(self, config: MMRConfig, input_chunk_count: int) -> dict:
        return {
            "enabled": config.enabled,
            "input_chunk_count": input_chunk_count,
            "selected_chunk_count": input_chunk_count,
            "removed_chunk_count": 0,
            "lambda": config.lambda_value,
            "top_k": config.top_k,
            "min_score": config.min_score,
            "similarity_threshold": config.similarity_threshold,
            "failed": False,
            "error": None,
        }
