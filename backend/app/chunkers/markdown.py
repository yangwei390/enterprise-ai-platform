from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.config.settings import settings
from backend.app.documents.metadata import ChunkMetadataBuilder
from backend.app.documents.router import StructureParserFactory
from backend.app.documents.schemas import DocumentStructure, DocumentStructureNode


class MarkdownChunker(BaseChunker):
    strategy = "markdown"

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        structure = source_metadata.get("_document_structure")
        if not isinstance(structure, DocumentStructure):
            structure = StructureParserFactory.parse(text, source_metadata, "markdown")
        chunks = self.chunk_structure(structure, source_metadata)
        return ChunkResult(
            strategy=self.strategy,
            chunk_size=settings.CHUNK_MARKDOWN_MAX_CHARS,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata={
                **source_metadata,
                "strategy": self.strategy,
                "chunk_strategy": self.strategy,
                "document_structure": structure.metadata,
            },
        )

    def chunk_structure(self, structure: DocumentStructure, source_metadata: dict) -> list[Chunk]:
        heading_nodes = [node for node in structure.nodes if node.node_type == "heading"]
        if not heading_nodes:
            return RecursiveChunker(chunk_size=settings.CHUNK_MARKDOWN_MAX_CHARS).chunk(
                structure.nodes[0].text if structure.nodes else "",
                {**source_metadata, "document_type": "markdown"},
            ).chunks
        chunks: list[Chunk] = []
        builder = ChunkMetadataBuilder()
        for node in heading_nodes:
            chunks.extend(
                self._node_to_chunks(
                    node=node,
                    source_metadata=source_metadata,
                    start_index=len(chunks),
                    builder=builder,
                )
            )
        return _reindex_chunks(chunks)

    def _node_to_chunks(
        self,
        *,
        node: DocumentStructureNode,
        source_metadata: dict,
        start_index: int,
        builder: ChunkMetadataBuilder,
    ) -> list[Chunk]:
        text = f"{'#' * node.level} {node.title}\n{node.text}".strip()
        if not text:
            return []
        metadata = {
            "document_type": "markdown",
            "heading_level": node.metadata.get("heading_level") or node.level,
            "heading_title": node.title,
            "parent_section": node.path[-2] if len(node.path) > 1 else None,
            "section_path": node.path,
            "structure_node_ids": [node.id],
        }
        if len(text) <= settings.CHUNK_MARKDOWN_MAX_CHARS:
            chunk = Chunk(
                document_id=source_metadata.get("document_id"),
                knowledge_base_id=source_metadata.get("knowledge_base_id"),
                chunk_index=start_index,
                text=text,
                start_offset=node.start_offset or 0,
                end_offset=node.end_offset or (node.start_offset or 0) + len(text),
                token_count=len(text),
                metadata={},
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata={**source_metadata, "document_type": "markdown"},
                structure_metadata=metadata,
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=node.level,
            )
            return [chunk]

        recursive = RecursiveChunker(
            chunk_size=settings.CHUNK_MARKDOWN_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
        ).chunk(text, {**source_metadata, "document_type": "markdown", **metadata})
        result = []
        for index, recursive_chunk in enumerate(recursive.chunks):
            chunk = recursive_chunk.model_copy(update={"chunk_index": start_index + index})
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata={**source_metadata, "document_type": "markdown"},
                structure_metadata=metadata,
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=node.level,
            )
            result.append(chunk)
        return result


def _reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    result = []
    for index, chunk in enumerate(chunks):
        result.append(
            chunk.model_copy(
                update={"chunk_index": index, "metadata": {**chunk.metadata, "chunk_index": index}}
            )
        )
    return result
