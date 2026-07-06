from typing import Any

from backend.app.context.compression.base import ContextCompressionResult


class SimpleContextCompressor:
    def compress(
        self,
        context_text: str,
        chunks: list[Any],
        max_chars: int = 6000,
    ) -> ContextCompressionResult:
        original_chars = len(context_text)
        original_chunk_count = len(chunks)

        if original_chars <= max_chars:
            return ContextCompressionResult(
                context_text=context_text,
                chunks=chunks,
                original_chars=original_chars,
                compressed_chars=original_chars,
                compression_applied=False,
                metadata={
                    "max_chars": max_chars,
                    "original_chunk_count": original_chunk_count,
                    "compressed_chunk_count": original_chunk_count,
                    "skipped_chunk_count": 0,
                },
            )

        compressed_chunks = []
        context_parts: list[str] = []
        current_chars = 0

        for chunk in chunks:
            chunk_text = chunk.text
            separator_chars = 2 if context_parts else 0
            next_chars = current_chars + separator_chars + len(chunk_text)
            if next_chars > max_chars:
                break

            compressed_chunks.append(chunk)
            context_parts.append(chunk_text)
            current_chars = next_chars

        compressed_context = "\n\n".join(context_parts)
        return ContextCompressionResult(
            context_text=compressed_context,
            chunks=compressed_chunks,
            original_chars=original_chars,
            compressed_chars=len(compressed_context),
            compression_applied=True,
            metadata={
                "max_chars": max_chars,
                "original_chunk_count": original_chunk_count,
                "compressed_chunk_count": len(compressed_chunks),
                "skipped_chunk_count": original_chunk_count - len(compressed_chunks),
            },
        )
