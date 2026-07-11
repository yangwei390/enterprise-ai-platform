import hashlib
import json
from typing import Any

from backend.app.chunkers.base import Chunk


class ChunkMetadataBuilder:
    base_keys = [
        "source",
        "document_id",
        "knowledge_base_id",
        "document_type",
        "parser",
        "cleaner",
        "chunk_strategy",
        "chunk_index",
        "chunk_uid",
        "chunk_role",
        "chunk_level",
        "parent_chunk_id",
        "section_path",
        "structure_node_ids",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
    ]

    def build(
        self,
        *,
        chunk: Chunk,
        source_metadata: dict,
        structure_metadata: dict | None = None,
        strategy: str,
        chunk_role: str = "child",
        chunk_level: int = 1,
        parent_chunk_id: str | None = None,
        child_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        extra = structure_metadata or {}
        section_path_raw = extra.get("section_path") or source_metadata.get("section_path") or []
        section_path = (
            [str(item) for item in section_path_raw]
            if isinstance(section_path_raw, list)
            else []
        )
        chunk_uid = str(
            extra.get("chunk_uid")
            or self.build_chunk_uid(
                document_id=chunk.document_id,
                section_path=section_path,
                text=chunk.text,
                article_start=extra.get("article_start"),
                article_end=extra.get("article_end"),
            )
        )
        metadata: dict[str, Any] = {
            **source_metadata,
            **extra,
            "source": source_metadata.get("source"),
            "document_id": chunk.document_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "document_type": extra.get("document_type") or source_metadata.get("document_type"),
            "parser": source_metadata.get("parser"),
            "cleaner": source_metadata.get("cleaner"),
            "chunk_strategy": strategy,
            "strategy": strategy,
            "chunk_index": chunk.chunk_index,
            "chunk_uid": chunk_uid,
            "chunk_id": chunk_uid,
            "chunk_role": chunk_role,
            "chunk_level": chunk_level,
            "parent_chunk_id": parent_chunk_id,
            "child_chunk_ids": child_chunk_ids or extra.get("child_chunk_ids") or [],
            "section_path": section_path,
            "structure_node_ids": _json_safe(extra.get("structure_node_ids") or []),
            "char_start": chunk.start_offset,
            "char_end": chunk.end_offset,
        }
        return _json_safe(metadata)

    def build_chunk_uid(
        self,
        *,
        document_id: int | None,
        section_path: list[str],
        text: str,
        article_start: Any | None = None,
        article_end: Any | None = None,
    ) -> str:
        normalized_text = " ".join(text.split())
        raw = json.dumps(
            {
                "document_id": document_id,
                "section_path": section_path,
                "article_start": article_start,
                "article_end": article_end,
                "text_hash": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set):
            return [_json_safe(item) for item in value]
        return str(value)
