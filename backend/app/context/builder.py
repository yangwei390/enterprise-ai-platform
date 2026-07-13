from backend.app.context.base import (
    BaseContextBuilder,
    ContextBuildRequest,
    ContextBuildResult,
    ContextChunk,
)


class BasicContextBuilder(BaseContextBuilder):
    def build(self, request: ContextBuildRequest) -> ContextBuildResult:
        seen_ids: set[str] = set()
        context_parts: list[str] = []
        context_chunks: list[ContextChunk] = []
        skipped_chunks = 0

        sorted_chunks = sorted(
            request.chunks,
            key=lambda chunk: chunk.rerank_score,
            reverse=True,
        )

        for chunk in sorted_chunks:
            if chunk.id in seen_ids:
                skipped_chunks += 1
                continue

            source = chunk.metadata.get("source")
            chunk_text = self._format_chunk(chunk, source)
            next_context = "\n\n".join([*context_parts, chunk_text])
            if len(next_context) > request.max_context_chars:
                skipped_chunks += 1
                continue

            seen_ids.add(chunk.id)
            context_parts.append(chunk_text)
            context_chunks.append(
                ContextChunk(
                    id=chunk.id,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    score=chunk.rerank_score,
                    source=source,
                    metadata=chunk.metadata,
                )
            )

        context_text = "\n\n".join(context_parts)
        return ContextBuildResult(
            query=request.query,
            context_text=context_text,
            chunks=context_chunks,
            total_chunks=len(context_chunks),
            total_chars=len(context_text),
            metadata={
                "max_context_chars": request.max_context_chars,
                "used_chunks": len(context_chunks),
                "skipped_chunks": skipped_chunks,
            },
        )

    def _format_chunk(self, chunk, source: str | None) -> str:
        metadata_lines = self._format_structure_metadata(chunk.metadata)
        header_lines = [
            f"Document: {source}",
            f"Document ID: {chunk.document_id}",
            f"Chunk: {chunk.chunk_index}",
            *metadata_lines,
            "",
            "正文:",
            chunk.text,
        ]
        return "\n".join(header_lines)

    def _format_structure_metadata(self, metadata: dict) -> list[str]:
        lines: list[str] = []
        structure_fields = [
            ("section_path", "Section Path"),
            ("chapter_label", "Chapter"),
            ("chapter_title", "Chapter Title"),
            ("article_label", "Article"),
        ]
        for key, label in structure_fields:
            value = metadata.get(key)
            if value in (None, "", []):
                continue
            if key == "section_path":
                value = self._format_section_path(value)
            lines.append(f"{label}: {value}")
        return lines

    def _format_section_path(self, value) -> str:
        if isinstance(value, list):
            return " > ".join(str(item) for item in value if item not in (None, ""))
        return str(value)
