from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.config.settings import settings
from backend.app.documents.metadata import ChunkMetadataBuilder
from backend.app.documents.router import StructureParserFactory
from backend.app.documents.schemas import DocumentStructure, DocumentStructureNode


class LegalStructureChunker(BaseChunker):
    strategy = "legal_structure"

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        structure = source_metadata.get("_document_structure")
        if not isinstance(structure, DocumentStructure):
            structure = StructureParserFactory.parse(text, source_metadata, "legal")
        chunks = self.chunk_structure(structure, source_metadata)
        return ChunkResult(
            strategy=self.strategy,
            chunk_size=settings.CHUNK_LEGAL_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
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

    def chunk_structure(
        self,
        structure: DocumentStructure,
        source_metadata: dict,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        builder = ChunkMetadataBuilder()
        chapter_nodes = [node for node in structure.nodes if node.node_type == "chapter"]
        article_nodes = [node for node in structure.nodes if node.node_type == "article"]
        for chapter in chapter_nodes:
            chapter_articles = [
                article
                for article in article_nodes
                if article.start_offset is not None
                and chapter.start_offset is not None
                and chapter.end_offset is not None
                and chapter.start_offset <= article.start_offset < chapter.end_offset
            ]
            parent_uid = builder.build_chunk_uid(
                document_id=source_metadata.get("document_id"),
                section_path=chapter.path,
                text=chapter.text,
            )
            if not chapter_articles:
                chunks.extend(
                    self._split_node(
                        node=chapter,
                        source_metadata=source_metadata,
                        start_index=len(chunks),
                        parent_uid=parent_uid,
                        builder=builder,
                    )
                )
                continue
            for article in chapter_articles:
                chunks.extend(
                    self._split_node(
                        node=article,
                        source_metadata=source_metadata,
                        start_index=len(chunks),
                        parent_uid=parent_uid,
                        builder=builder,
                    )
                )
        if not chunks:
            recursive_metadata = {**source_metadata, "document_type": "legal"}
            return RecursiveChunker(chunk_size=settings.CHUNK_LEGAL_MAX_CHARS).chunk(
                structure.nodes[0].text if structure.nodes else "",
                recursive_metadata,
            ).chunks
        return _reindex_chunks(chunks)

    def _split_node(
        self,
        *,
        node: DocumentStructureNode,
        source_metadata: dict,
        start_index: int,
        parent_uid: str,
        builder: ChunkMetadataBuilder,
    ) -> list[Chunk]:
        node_text = node.text.strip()
        if not node_text:
            return []
        if len(node_text) <= settings.CHUNK_LEGAL_MAX_CHARS:
            chunk = Chunk(
                document_id=source_metadata.get("document_id"),
                knowledge_base_id=source_metadata.get("knowledge_base_id"),
                chunk_index=start_index,
                text=node_text,
                start_offset=node.start_offset or 0,
                end_offset=node.end_offset or (node.start_offset or 0) + len(node_text),
                token_count=len(node_text),
                metadata={},
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata={**source_metadata, "document_type": "legal"},
                structure_metadata=_legal_metadata(node),
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=node.level,
                parent_chunk_id=parent_uid,
            )
            return [chunk]

        recursive_result = RecursiveChunker(
            chunk_size=settings.CHUNK_LEGAL_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
        ).chunk(node_text, {**source_metadata, "document_type": "legal"})
        split_chunks = []
        for index, recursive_chunk in enumerate(recursive_result.chunks):
            start = (node.start_offset or 0) + recursive_chunk.start_offset
            end = (node.start_offset or 0) + recursive_chunk.end_offset
            chunk = recursive_chunk.model_copy(
                update={
                    "chunk_index": start_index + index,
                    "start_offset": start,
                    "end_offset": end,
                }
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata={**source_metadata, "document_type": "legal"},
                structure_metadata=_legal_metadata(node),
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=node.level,
                parent_chunk_id=parent_uid,
            )
            split_chunks.append(chunk)
        return split_chunks


def _legal_metadata(node: DocumentStructureNode) -> dict:
    metadata = {
        **node.metadata,
        "document_type": "legal",
        "section_path": node.metadata.get("section_path") or node.path,
        "structure_node_ids": [node.id],
    }
    article_number = metadata.get("article_number")
    article_label = metadata.get("article_label")
    if article_number is not None:
        metadata["article_start"] = article_number
        metadata["article_end"] = article_number
        metadata["article_labels"] = [article_label] if article_label else []
    return metadata


def _reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    result = []
    for index, chunk in enumerate(chunks):
        metadata = {**chunk.metadata, "chunk_index": index}
        result.append(chunk.model_copy(update={"chunk_index": index, "metadata": metadata}))
    return result
