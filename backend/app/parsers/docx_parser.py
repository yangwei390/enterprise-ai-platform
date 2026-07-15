from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from backend.app.parsers.base import (
    BaseParser,
    DocumentElement,
    ParseQuality,
    ParseResult,
    TableStructure,
)

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxParser(BaseParser):
    def parse(self, file_path: Path) -> ParseResult:
        document_xml = self._read_document_xml(file_path)
        root = ElementTree.fromstring(document_xml)
        body = root.find(f"{WORD_NS}body")
        elements: list[DocumentElement] = []
        text_parts: list[str] = []
        section_path: list[str] = []

        if body is not None:
            for child in body:
                if child.tag == f"{WORD_NS}p":
                    content = _paragraph_text(child)
                    if not content:
                        continue
                    element_type = _paragraph_type(child, content)
                    if element_type == "heading":
                        section_path = [content]
                    elements.append(
                        DocumentElement(
                            type=element_type,
                            content=content,
                            section_path=list(section_path),
                        )
                    )
                    text_parts.append(content)
                elif child.tag == f"{WORD_NS}tbl":
                    table = _parse_table(child)
                    if table is None:
                        continue
                    elements.append(
                        DocumentElement(
                            type="table",
                            content=_table_to_text(table),
                            section_path=list(section_path),
                            metadata={"table": table.model_dump()},
                        )
                    )
                    text_parts.append(_table_to_text(table))

        text = "\n".join(text_parts)
        table_count = sum(1 for element in elements if element.type == "table")
        return ParseResult(
            text=text,
            page_count=None,
            metadata={
                "suffix": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
            },
            elements=elements,
            parse_quality=ParseQuality(
                element_count=len(elements),
                table_count=table_count,
            ),
        )

    def _read_document_xml(self, file_path: Path) -> bytes:
        with ZipFile(file_path) as archive:
            return archive.read("word/document.xml")


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    texts = [
        node.text or ""
        for node in paragraph.iter(f"{WORD_NS}t")
        if node.text is not None
    ]
    return "".join(texts).strip()


def _paragraph_type(paragraph: ElementTree.Element, content: str) -> str:
    style_value = None
    paragraph_properties = paragraph.find(f"{WORD_NS}pPr")
    if paragraph_properties is not None:
        style = paragraph_properties.find(f"{WORD_NS}pStyle")
        if style is not None:
            style_value = style.attrib.get(f"{WORD_NS}val")
        numbering = paragraph_properties.find(f"{WORD_NS}numPr")
        if numbering is not None:
            return "list"
    if style_value and style_value.lower().startswith("heading"):
        return "heading"
    if content.startswith(("- ", "* ", "• ")):
        return "list"
    return "paragraph"


def _parse_table(table_node: ElementTree.Element) -> TableStructure | None:
    rows: list[list[str]] = []
    for row_node in table_node.findall(f"{WORD_NS}tr"):
        row = []
        for cell_node in row_node.findall(f"{WORD_NS}tc"):
            cell_text = " ".join(
                filter(
                    None,
                    (_paragraph_text(paragraph) for paragraph in cell_node.findall(f"{WORD_NS}p")),
                )
            )
            row.append(cell_text.strip())
        if any(row):
            rows.append(row)
    if not rows:
        return None
    headers = rows[0]
    return TableStructure(
        headers=headers,
        rows=rows[1:],
        units=_extract_units(" ".join(headers)),
    )


def _table_to_text(table: TableStructure) -> str:
    lines = []
    if table.title:
        lines.append(table.title)
    if table.headers:
        lines.append(" | ".join(table.headers))
    lines.extend(" | ".join(row) for row in table.rows)
    return "\n".join(lines)


def _extract_units(text: str) -> str | None:
    for marker in ("单位：", "单位:"):
        if marker in text:
            return text.split(marker, 1)[1].strip() or None
    return None
