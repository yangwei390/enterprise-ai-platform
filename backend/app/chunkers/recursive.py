from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult
from backend.app.config.settings import settings
from backend.app.documents.metadata import ChunkMetadataBuilder


class RecursiveChunker(BaseChunker):
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chars: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_RECURSIVE_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_RECURSIVE_OVERLAP
        self.min_chars = min_chars or settings.CHUNK_MIN_CHARS

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        pieces = self.split_text(text)
        chunks: list[Chunk] = []
        offset = 0
        builder = ChunkMetadataBuilder()
        for index, piece in enumerate(pieces):
            start = text.find(piece, offset)
            if start < 0:
                start = offset
            end = start + len(piece)
            chunk = Chunk(
                document_id=source_metadata.get("document_id"),
                knowledge_base_id=source_metadata.get("knowledge_base_id"),
                chunk_index=index,
                text=piece,
                start_offset=start,
                end_offset=end,
                token_count=len(piece),
                metadata={},
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata=source_metadata,
                strategy="recursive",
                structure_metadata={
                    "document_type": source_metadata.get("document_type") or "plain_text",
                    "section_path": source_metadata.get("section_path") or [],
                },
            )
            chunks.append(chunk)
            offset = max(end - self.chunk_overlap, end)
        return ChunkResult(
            strategy="recursive",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata={
                **source_metadata,
                "strategy": "recursive",
                "chunk_strategy": "recursive",
            },
        )

    def split_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]
        chunks: list[str] = []
        current = normalized
        while current:
            if len(current) <= self.chunk_size:
                chunks.append(current)
                break
            split_at = self._find_split(current)
            piece = current[:split_at].strip()
            if piece:
                chunks.append(piece)
            overlap_text = piece[-self.chunk_overlap :] if self.chunk_overlap else ""
            current = (overlap_text + current[split_at:]).strip()
            if len(chunks) > 10000:
                break
        return self._merge_tiny_chunks(chunks)

    def _find_split(self, text: str) -> int:
        window = text[: self.chunk_size]
        for separator in self.separators:
            position = window.rfind(separator)
            if position >= self.min_chars:
                return position + len(separator)
        return self.chunk_size

    def _merge_tiny_chunks(self, chunks: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            if len(chunk) < self.min_chars and buffer:
                buffer = f"{buffer}{chunk}"
                continue
            if buffer:
                merged.append(buffer)
            buffer = chunk
        if buffer:
            merged.append(buffer)
        return merged
