from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.config.settings import settings
from backend.app.documents.metadata import ChunkMetadataBuilder


class ParentChildChunker(BaseChunker):
    strategy = "parent_child"

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        builder = ChunkMetadataBuilder()
        parent_uid = builder.build_chunk_uid(
            document_id=source_metadata.get("document_id"),
            section_path=source_metadata.get("section_path") or [],
            text=text[: settings.CHUNK_PARENT_MAX_CHARS],
        )
        parent_chunk = Chunk(
            document_id=source_metadata.get("document_id"),
            knowledge_base_id=source_metadata.get("knowledge_base_id"),
            chunk_index=0,
            text=text[: settings.CHUNK_PARENT_MAX_CHARS],
            start_offset=0,
            end_offset=min(len(text), settings.CHUNK_PARENT_MAX_CHARS),
            token_count=min(len(text), settings.CHUNK_PARENT_MAX_CHARS),
            metadata={},
        )
        child_result = RecursiveChunker(
            chunk_size=settings.CHUNK_CHILD_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
        ).chunk(text, source_metadata)
        child_chunks = []
        start_index = 1 if settings.CHUNK_EMBED_PARENT else 0
        for index, child in enumerate(child_result.chunks, start=start_index):
            updated = child.model_copy(update={"chunk_index": index})
            updated.metadata = builder.build(
                chunk=updated,
                source_metadata=source_metadata,
                structure_metadata={
                    "document_type": source_metadata.get("document_type") or "plain_text",
                    "section_path": source_metadata.get("section_path") or [],
                },
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=1,
                parent_chunk_id=parent_uid,
            )
            child_chunks.append(updated)

        parent_chunk.metadata = builder.build(
            chunk=parent_chunk,
            source_metadata=source_metadata,
            strategy=self.strategy,
            chunk_role="parent",
            chunk_level=0,
            child_chunk_ids=[chunk.metadata.get("chunk_uid") for chunk in child_chunks],
        )
        chunks = [parent_chunk, *child_chunks] if settings.CHUNK_EMBED_PARENT else child_chunks
        return ChunkResult(
            strategy=self.strategy,
            chunk_size=settings.CHUNK_CHILD_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata={
                **source_metadata,
                "strategy": self.strategy,
                "parent_count": 1,
                "child_count": len(child_chunks),
            },
        )
