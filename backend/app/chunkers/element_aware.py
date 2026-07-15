from typing import Any

from backend.app.chunkers.base import Chunk, ChunkResult
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.config.settings import settings
from backend.app.documents.metadata import ChunkMetadataBuilder
from pydantic import BaseModel, Field


class ParsedElement(BaseModel):
    type: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class ElementAwareChunker:
    strategy = "parent_child"

    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        source_metadata = metadata or {}
        elements = _coerce_elements(source_metadata.get("_parser_elements"))
        if not elements:
            return RecursiveChunker(
                chunk_size=settings.CHUNK_CHILD_MAX_CHARS,
                chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
            ).chunk(text, source_metadata)

        builder = ChunkMetadataBuilder()
        chunks: list[Chunk] = []
        for section in _group_sections(elements):
            chunks.extend(
                self._section_to_chunks(
                    section=section,
                    source_metadata=source_metadata,
                    start_index=len(chunks),
                    builder=builder,
                )
            )
        chunks = _reindex_chunks(chunks)
        return ChunkResult(
            strategy=self.strategy,
            chunk_size=settings.CHUNK_CHILD_MAX_CHARS,
            chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count or 0 for chunk in chunks),
            metadata={
                **_public_metadata(source_metadata),
                "strategy": self.strategy,
                "chunk_strategy": self.strategy,
                "structure_source": "parser_elements",
                "parser_element_count": len(elements),
                "parent_count": sum(
                    1 for chunk in chunks if chunk.metadata.get("chunk_role") == "parent"
                ),
                "child_count": sum(
                    1 for chunk in chunks if chunk.metadata.get("chunk_role") == "child"
                ),
            },
        )

    def _section_to_chunks(
        self,
        *,
        section: list[ParsedElement],
        source_metadata: dict,
        start_index: int,
        builder: ChunkMetadataBuilder,
    ) -> list[Chunk]:
        if not section:
            return []
        section_path = _section_path(section)
        heading = next((element.content for element in section if element.type == "heading"), None)
        parent_text = "\n".join(element.content for element in section if element.content).strip()
        parent_uid = builder.build_chunk_uid(
            document_id=source_metadata.get("document_id"),
            section_path=section_path,
            text=parent_text[: settings.CHUNK_PARENT_MAX_CHARS],
        )
        child_chunks = self._build_child_chunks(
            section=section,
            source_metadata=source_metadata,
            start_index=start_index + (1 if settings.CHUNK_EMBED_PARENT else 0),
            parent_uid=parent_uid,
            builder=builder,
            heading_title=heading,
            section_path=section_path,
        )
        if not settings.CHUNK_EMBED_PARENT:
            return child_chunks

        parent_chunk = Chunk(
            document_id=source_metadata.get("document_id"),
            knowledge_base_id=source_metadata.get("knowledge_base_id"),
            chunk_index=start_index,
            text=parent_text[: settings.CHUNK_PARENT_MAX_CHARS],
            start_offset=0,
            end_offset=min(len(parent_text), settings.CHUNK_PARENT_MAX_CHARS),
            token_count=min(len(parent_text), settings.CHUNK_PARENT_MAX_CHARS),
            metadata={},
        )
        parent_chunk.metadata = builder.build(
            chunk=parent_chunk,
            source_metadata=_public_metadata(source_metadata),
            structure_metadata={
                "document_type": source_metadata.get("document_type") or "plain_text",
                "element_types": _element_types(section),
                "page_start": _page_start(section),
                "page_end": _page_end(section),
                "section_path": section_path,
                "heading_title": heading,
                "structure_source": "parser_elements",
            },
            strategy=self.strategy,
            chunk_role="parent",
            chunk_level=0,
            child_chunk_ids=[
                str(chunk.metadata["chunk_uid"])
                for chunk in child_chunks
                if chunk.metadata.get("chunk_uid")
            ],
        )
        return [parent_chunk, *child_chunks]

    def _build_child_chunks(
        self,
        *,
        section: list[ParsedElement],
        source_metadata: dict,
        start_index: int,
        parent_uid: str,
        builder: ChunkMetadataBuilder,
        heading_title: str | None,
        section_path: list[str],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[ParsedElement] = []
        for element in section:
            if element.type == "heading":
                buffer.append(element)
                continue
            if element.type == "table":
                chunks.extend(
                    self._flush_text_buffer(
                        buffer=buffer,
                        chunks=chunks,
                        source_metadata=source_metadata,
                        start_index=start_index,
                        parent_uid=parent_uid,
                        builder=builder,
                        heading_title=heading_title,
                        section_path=section_path,
                    )
                )
                buffer = []
                chunks.extend(
                    self._table_to_chunks(
                        element=element,
                        source_metadata=source_metadata,
                        start_index=start_index + len(chunks),
                        parent_uid=parent_uid,
                        builder=builder,
                        heading_title=heading_title,
                        section_path=section_path,
                    )
                )
                continue
            candidate = [*buffer, element]
            if _elements_length(candidate) > settings.CHUNK_CHILD_MAX_CHARS and buffer:
                chunks.extend(
                    self._flush_text_buffer(
                        buffer=buffer,
                        chunks=chunks,
                        source_metadata=source_metadata,
                        start_index=start_index,
                        parent_uid=parent_uid,
                        builder=builder,
                        heading_title=heading_title,
                        section_path=section_path,
                    )
                )
                buffer = [element]
            else:
                buffer = candidate
        chunks.extend(
            self._flush_text_buffer(
                buffer=buffer,
                chunks=chunks,
                source_metadata=source_metadata,
                start_index=start_index,
                parent_uid=parent_uid,
                builder=builder,
                heading_title=heading_title,
                section_path=section_path,
            )
        )
        return chunks

    def _flush_text_buffer(
        self,
        *,
        buffer: list[ParsedElement],
        chunks: list[Chunk],
        source_metadata: dict,
        start_index: int,
        parent_uid: str,
        builder: ChunkMetadataBuilder,
        heading_title: str | None,
        section_path: list[str],
    ) -> list[Chunk]:
        if not buffer:
            return []
        text = _element_group_text(buffer, heading_title)
        if not text:
            return []
        metadata = {
            "document_type": source_metadata.get("document_type") or "plain_text",
            "element_types": _element_types(buffer),
            "page_start": _page_start(buffer),
            "page_end": _page_end(buffer),
            "section_path": section_path,
            "heading_title": heading_title,
            "structure_source": "parser_elements",
        }
        return self._split_text_group(
            text=text,
            elements=buffer,
            source_metadata=source_metadata,
            start_index=start_index + len(chunks),
            parent_uid=parent_uid,
            builder=builder,
            structure_metadata=metadata,
        )

    def _split_text_group(
        self,
        *,
        text: str,
        elements: list[ParsedElement],
        source_metadata: dict,
        start_index: int,
        parent_uid: str,
        builder: ChunkMetadataBuilder,
        structure_metadata: dict,
    ) -> list[Chunk]:
        pieces = (
            [text]
            if len(text) <= settings.CHUNK_CHILD_MAX_CHARS
            else RecursiveChunker(
                chunk_size=settings.CHUNK_CHILD_MAX_CHARS,
                chunk_overlap=settings.CHUNK_RECURSIVE_OVERLAP,
            ).split_text(text)
        )
        result: list[Chunk] = []
        offset = 0
        for index, piece in enumerate(pieces):
            start = text.find(piece, offset)
            if start < 0:
                start = offset
            end = start + len(piece)
            chunk = Chunk(
                document_id=source_metadata.get("document_id"),
                knowledge_base_id=source_metadata.get("knowledge_base_id"),
                chunk_index=start_index + index,
                text=piece,
                start_offset=start,
                end_offset=end,
                token_count=len(piece),
                metadata={},
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata=_public_metadata(source_metadata),
                structure_metadata=structure_metadata,
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=1,
                parent_chunk_id=parent_uid,
            )
            result.append(chunk)
            offset = end
        return result

    def _table_to_chunks(
        self,
        *,
        element: ParsedElement,
        source_metadata: dict,
        start_index: int,
        parent_uid: str,
        builder: ChunkMetadataBuilder,
        heading_title: str | None,
        section_path: list[str],
    ) -> list[Chunk]:
        table = element.metadata.get("table") if isinstance(element.metadata, dict) else None
        if not isinstance(table, dict):
            return self._split_text_group(
                text=element.content,
                elements=[element],
                source_metadata=source_metadata,
                start_index=start_index,
                parent_uid=parent_uid,
                builder=builder,
                structure_metadata=_element_metadata(element, heading_title, section_path),
            )
        groups = _table_row_groups(table)
        result: list[Chunk] = []
        for index, rows in enumerate(groups):
            chunk_text = _table_group_text(table, rows)
            chunk = Chunk(
                document_id=source_metadata.get("document_id"),
                knowledge_base_id=source_metadata.get("knowledge_base_id"),
                chunk_index=start_index + index,
                text=chunk_text,
                start_offset=0,
                end_offset=len(chunk_text),
                token_count=len(chunk_text),
                metadata={},
            )
            chunk.metadata = builder.build(
                chunk=chunk,
                source_metadata=_public_metadata(source_metadata),
                structure_metadata={
                    **_element_metadata(element, heading_title, section_path),
                    "table_title": table.get("title"),
                    "table_headers": table.get("headers") or [],
                    "table_units": table.get("units"),
                    "table_continuation": table.get("continuation_of"),
                },
                strategy=self.strategy,
                chunk_role="child",
                chunk_level=1,
                parent_chunk_id=parent_uid,
            )
            result.append(chunk)
        return result


def _coerce_elements(raw_elements: Any) -> list[ParsedElement]:
    if not isinstance(raw_elements, list):
        return []
    elements: list[ParsedElement] = []
    for item in raw_elements:
        raw = item.model_dump() if hasattr(item, "model_dump") else item
        if not isinstance(raw, dict):
            continue
        element = ParsedElement(**raw)
        if element.content.strip():
            elements.append(element)
    return elements


def _group_sections(elements: list[ParsedElement]) -> list[list[ParsedElement]]:
    sections: list[list[ParsedElement]] = []
    current: list[ParsedElement] = []
    for element in elements:
        if element.type == "heading" and current:
            sections.append(current)
            current = [element]
        else:
            current.append(element)
    if current:
        sections.append(current)
    return sections


def _section_path(section: list[ParsedElement]) -> list[str]:
    for element in section:
        if element.section_path:
            return element.section_path
    heading = next((element.content for element in section if element.type == "heading"), None)
    return [heading] if heading else []


def _element_group_text(elements: list[ParsedElement], heading_title: str | None) -> str:
    parts = [element.content.strip() for element in elements if element.content.strip()]
    if heading_title and parts and parts[0] != heading_title:
        parts.insert(0, heading_title)
    return "\n".join(parts).strip()


def _elements_length(elements: list[ParsedElement]) -> int:
    return sum(len(element.content) for element in elements) + max(len(elements) - 1, 0)


def _element_types(elements: list[ParsedElement]) -> list[str]:
    return sorted({element.type for element in elements})


def _page_start(elements: list[ParsedElement]) -> int | None:
    values = [element.page_start for element in elements if element.page_start is not None]
    return min(values) if values else None


def _page_end(elements: list[ParsedElement]) -> int | None:
    values = [element.page_end for element in elements if element.page_end is not None]
    return max(values) if values else None


def _element_metadata(
    element: ParsedElement,
    heading_title: str | None,
    section_path: list[str],
) -> dict:
    return {
        "document_type": "structured",
        "element_types": [element.type],
        "page_start": element.page_start,
        "page_end": element.page_end,
        "section_path": element.section_path or section_path,
        "heading_title": heading_title,
        "structure_source": "parser_elements",
    }


def _table_row_groups(table: dict) -> list[list[list[str]]]:
    rows = table.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return [[]]
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in rows:
        row_values = [str(cell) for cell in row] if isinstance(row, list) else [str(row)]
        candidate = [*current, row_values]
        if len(_table_group_text(table, candidate)) > settings.CHUNK_CHILD_MAX_CHARS and current:
            groups.append(current)
            current = [row_values]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _table_group_text(table: dict, rows: list[list[str]]) -> str:
    lines = []
    title = table.get("title")
    headers = table.get("headers") or []
    units = table.get("units")
    if title:
        lines.append(str(title))
    if units and (not title or str(units) not in str(title)):
        lines.append(f"Unit: {units}")
    if headers:
        lines.append(" | ".join(str(header) for header in headers))
    lines.extend(" | ".join(str(cell) for cell in row) for row in rows)
    return "\n".join(lines).strip()


def _public_metadata(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if not key.startswith("_")}


def _reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    result = []
    for index, chunk in enumerate(chunks):
        result.append(
            chunk.model_copy(
                update={"chunk_index": index, "metadata": {**chunk.metadata, "chunk_index": index}}
            )
        )
    return result
