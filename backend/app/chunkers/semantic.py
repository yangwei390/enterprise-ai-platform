from math import sqrt

from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.config.settings import settings


class SemanticChunker(BaseChunker):
    strategy = "semantic"

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        if not settings.CHUNK_SEMANTIC_ENABLED:
            return self._fallback(text, source_metadata, "semantic disabled")
        paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
        if not paragraphs or len(paragraphs) > settings.CHUNK_SEMANTIC_MAX_PARAGRAPHS:
            return self._fallback(text, source_metadata, "paragraph limit")
        try:
            from backend.app.embeddings import EmbeddingFactory

            embedding = EmbeddingFactory.get_embedding()
            vectors = [embedding.embed_text(paragraph) for paragraph in paragraphs]
        except Exception as exc:
            return self._fallback(text, source_metadata, str(exc))

        groups: list[list[str]] = [[]]
        for index, paragraph in enumerate(paragraphs):
            if index == 0:
                groups[-1].append(paragraph)
                continue
            similarity = _cosine_similarity(vectors[index - 1], vectors[index])
            current_chars = sum(len(item) for item in groups[-1])
            if (
                similarity < settings.CHUNK_SEMANTIC_SIMILARITY_THRESHOLD
                or current_chars + len(paragraph) > settings.CHUNK_RECURSIVE_SIZE
            ):
                groups.append([paragraph])
            else:
                groups[-1].append(paragraph)

        recursive = RecursiveChunker()
        chunks: list[Chunk] = []
        for group in groups:
            chunks.extend(recursive.chunk("\n\n".join(group), source_metadata).chunks)
        chunks = [
            chunk.model_copy(
                update={
                    "chunk_index": index,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_strategy": self.strategy,
                        "strategy": self.strategy,
                        "semantic_fallback_used": False,
                    },
                }
            )
            for index, chunk in enumerate(chunks)
        ]
        return ChunkResult(
            strategy=self.strategy,
            chunk_size=settings.CHUNK_RECURSIVE_SIZE,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata={**source_metadata, "strategy": self.strategy},
        )

    def _fallback(self, text: str, metadata: dict, reason: str) -> ChunkResult:
        result = RecursiveChunker().chunk(text, metadata)
        return result.model_copy(
            update={
                "strategy": "recursive",
                "metadata": {
                    **result.metadata,
                    "fallback_used": True,
                    "fallback_reason": reason,
                    "fallback_chain": ["semantic", "recursive"],
                },
            }
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
