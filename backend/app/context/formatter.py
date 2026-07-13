from typing import Any

STRUCTURE_FIELDS = [
    ("section_path", "Section Path"),
    ("chapter_label", "Chapter"),
    ("chapter_title", "Chapter Title"),
    ("article_label", "Article"),
]


def format_context_chunks(chunks: list[Any]) -> str:
    return "\n\n".join(format_context_chunk(chunk) for chunk in chunks)


def format_context_chunk(chunk: Any) -> str:
    metadata = _chunk_metadata(chunk)
    header_lines = [
        f"Document: {_chunk_source(chunk, metadata)}",
        f"Document ID: {_chunk_document_id(chunk)}",
        f"Chunk: {_chunk_index(chunk)}",
        *_format_structure_metadata(metadata),
        "",
        "正文:",
        _chunk_text(chunk),
    ]
    return "\n".join(header_lines)


def _format_structure_metadata(metadata: dict) -> list[str]:
    lines: list[str] = []
    for key, label in STRUCTURE_FIELDS:
        value = metadata.get(key)
        if value in (None, "", []):
            continue
        if key == "section_path":
            value = _format_section_path(value)
        lines.append(f"{label}: {value}")
    return lines


def _format_section_path(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def _chunk_metadata(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") or {}
    else:
        metadata = getattr(chunk, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _chunk_source(chunk: Any, metadata: dict) -> str | None:
    if isinstance(chunk, dict):
        source = chunk.get("source")
    else:
        source = getattr(chunk, "source", None)
    return source or metadata.get("source")


def _chunk_document_id(chunk: Any) -> int | None:
    if isinstance(chunk, dict):
        value = chunk.get("document_id")
    else:
        value = getattr(chunk, "document_id", None)
    return value if isinstance(value, int) else None


def _chunk_index(chunk: Any) -> int | None:
    if isinstance(chunk, dict):
        value = chunk.get("chunk_index")
    else:
        value = getattr(chunk, "chunk_index", None)
    return value if isinstance(value, int) else None
