from dataclasses import dataclass
from typing import Protocol

from backend.app.config.settings import settings
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.neighbor_lookup import QdrantNeighborChunkLookup
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class NeighborChunkLookup(Protocol):
    def find_neighbor(
        self,
        *,
        document_id: int,
        knowledge_base_id: int,
        chunk_index: int,
    ) -> RerankedChunk | None: ...


@dataclass
class NeighborExpansionConfig:
    enabled: bool = True
    before: int = 1
    after: int = 1
    max_added_chunks: int = 10
    fail_open: bool = True


def get_neighbor_expansion_config() -> NeighborExpansionConfig:
    return NeighborExpansionConfig(
        enabled=settings.NEIGHBOR_EXPANSION_ENABLED,
        before=settings.NEIGHBOR_EXPANSION_BEFORE,
        after=settings.NEIGHBOR_EXPANSION_AFTER,
        max_added_chunks=settings.NEIGHBOR_EXPANSION_MAX_ADDED_CHUNKS,
        fail_open=settings.NEIGHBOR_EXPANSION_FAIL_OPEN,
    )


class NeighborExpansionStep(BaseRetrieverStep):
    def __init__(self, lookup: NeighborChunkLookup | None = None) -> None:
        self.lookup = lookup or QdrantNeighborChunkLookup()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        config = get_neighbor_expansion_config()
        metadata = self._base_metadata(config, len(context.reranked_chunks))
        if not config.enabled:
            context.metadata["neighbor_expansion"] = metadata
            return context

        original_chunks = context.reranked_chunks
        try:
            expanded_chunks, added_count, skipped_count = self._expand_chunks(
                chunks=original_chunks,
                config=config,
            )
            context.reranked_chunks = expanded_chunks
            context.metadata["neighbor_expansion"] = {
                **metadata,
                "added_chunk_count": added_count,
                "output_chunk_count": len(expanded_chunks),
                "skipped_chunk_count": skipped_count,
                "failed": False,
                "error": None,
            }
            return context
        except Exception as exc:
            context.add_error("NeighborExpansionStep", exc)
            metadata["failed"] = True
            metadata["error"] = str(exc)
            metadata["output_chunk_count"] = len(original_chunks)
            context.metadata["neighbor_expansion"] = metadata
            if not config.fail_open:
                raise RuntimeError(f"Neighbor expansion failed: {exc}") from exc
            context.reranked_chunks = original_chunks
            return context

    def _expand_chunks(
        self,
        *,
        chunks: list[RerankedChunk],
        config: NeighborExpansionConfig,
    ) -> tuple[list[RerankedChunk], int, int]:
        expanded_chunks: list[RerankedChunk] = []
        seen_keys: set[tuple[int | None, int | None, int | None]] = set()
        added_count = 0
        skipped_count = 0

        for chunk in chunks:
            key = self._chunk_key(chunk)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if not self._can_expand(chunk):
                skipped_count += 1
                expanded_chunks.append(chunk)
                continue

            document_id = chunk.document_id
            knowledge_base_id = chunk.knowledge_base_id
            chunk_index = chunk.chunk_index
            if document_id is None or knowledge_base_id is None or chunk_index is None:
                skipped_count += 1
                expanded_chunks.append(chunk)
                continue

            before_chunks = self._lookup_neighbors(
                parent=chunk,
                indexes=range(
                    chunk_index - config.before,
                    chunk_index,
                ),
                position="before",
                seen_keys=seen_keys,
                max_remaining=config.max_added_chunks - added_count,
            )
            added_count += len(before_chunks)
            expanded_chunks.extend(before_chunks)

            expanded_chunks.append(chunk)

            after_chunks = self._lookup_neighbors(
                parent=chunk,
                indexes=range(
                    chunk_index + 1,
                    chunk_index + config.after + 1,
                ),
                position="after",
                seen_keys=seen_keys,
                max_remaining=config.max_added_chunks - added_count,
            )
            added_count += len(after_chunks)
            expanded_chunks.extend(after_chunks)

            if added_count >= config.max_added_chunks:
                remaining = [item for item in chunks if self._chunk_key(item) not in seen_keys]
                expanded_chunks.extend(remaining)
                break

        return expanded_chunks, added_count, skipped_count

    def _lookup_neighbors(
        self,
        *,
        parent: RerankedChunk,
        indexes: range,
        position: str,
        seen_keys: set[tuple[int | None, int | None, int | None]],
        max_remaining: int,
    ) -> list[RerankedChunk]:
        if max_remaining <= 0:
            return []
        if (
            parent.document_id is None
            or parent.knowledge_base_id is None
            or parent.chunk_index is None
        ):
            return []

        neighbors: list[RerankedChunk] = []
        for neighbor_index in indexes:
            if len(neighbors) >= max_remaining:
                break
            neighbor_key = (parent.knowledge_base_id, parent.document_id, neighbor_index)
            if neighbor_key in seen_keys:
                continue
            neighbor = self.lookup.find_neighbor(
                document_id=parent.document_id,
                knowledge_base_id=parent.knowledge_base_id,
                chunk_index=neighbor_index,
            )
            if neighbor is None:
                continue
            neighbor = self._mark_neighbor(
                neighbor=neighbor,
                parent=parent,
                position=position,
                distance=abs(neighbor_index - parent.chunk_index),
            )
            seen_keys.add(neighbor_key)
            neighbors.append(neighbor)
        return neighbors

    def _mark_neighbor(
        self,
        *,
        neighbor: RerankedChunk,
        parent: RerankedChunk,
        position: str,
        distance: int,
    ) -> RerankedChunk:
        metadata = {
            **neighbor.metadata,
            "neighbor_expanded": True,
            "neighbor_of_document_id": parent.document_id,
            "neighbor_of_chunk_index": parent.chunk_index,
            "neighbor_position": position,
            "neighbor_distance": distance,
            "neighbor_expansion_provider": "metadata_lookup",
            "inherited_from_rerank_rank": parent.metadata.get("rerank_rank"),
            "neighbor_parent_rerank_score": parent.rerank_score,
        }
        return neighbor.model_copy(update={"metadata": metadata})

    def _base_metadata(
        self,
        config: NeighborExpansionConfig,
        input_chunk_count: int,
    ) -> dict:
        return {
            "enabled": config.enabled,
            "before": config.before,
            "after": config.after,
            "input_chunk_count": input_chunk_count,
            "added_chunk_count": 0,
            "output_chunk_count": input_chunk_count,
            "skipped_chunk_count": 0,
            "failed": False,
            "error": None,
        }

    def _can_expand(self, chunk: RerankedChunk) -> bool:
        return (
            chunk.document_id is not None
            and chunk.knowledge_base_id is not None
            and chunk.chunk_index is not None
        )

    def _chunk_key(self, chunk: RerankedChunk) -> tuple[int | None, int | None, int | None]:
        return chunk.knowledge_base_id, chunk.document_id, chunk.chunk_index
