from backend.app.chunkers.base import BaseChunker, Chunk, ChunkResult


class FixedChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        result_metadata = {
            **source_metadata,
            "strategy": "fixed",
        }
        chunk_metadata = {
            "source": source_metadata.get("source"),
            "parser": source_metadata.get("parser"),
            "cleaner": source_metadata.get("cleaner"),
            "page_count": source_metadata.get("page_count"),
            "strategy": "fixed",
        }

        if not text:
            return ChunkResult(
                strategy="fixed",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunks=[],
                total_chunks=0,
                total_tokens=0,
                metadata=result_metadata,
            )

        chunks: list[Chunk] = []
        start_offset = 0
        chunk_index = 0
        text_length = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start_offset < text_length:
            end_offset = min(start_offset + self.chunk_size, text_length)
            chunk_text = text[start_offset:end_offset]
            chunks.append(
                Chunk(
                    document_id=source_metadata.get("document_id"),
                    knowledge_base_id=source_metadata.get("knowledge_base_id"),
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=len(chunk_text),
                    metadata=chunk_metadata,
                )
            )

            if end_offset >= text_length:
                break

            start_offset += step
            chunk_index += 1

        return ChunkResult(
            strategy="fixed",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata=result_metadata,
        )
