import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from backend.app.parsers.base import (
    BaseParser,
    DocumentElement,
    ParseQuality,
    ParseResult,
    TableStructure,
)


@dataclass
class PdfLine:
    text: str
    page_number: int
    bbox: list[float]
    font_size: float
    flags: int = 0


class PdfParser(BaseParser):
    def parse(self, file_path: Path) -> ParseResult:
        with fitz.open(file_path) as document:
            page_count = document.page_count
            pages = [
                self._extract_page_lines(document.load_page(index), index + 1)
                for index in range(page_count)
            ]

        repeated_header_footer = self._detect_repeated_header_footer(pages)
        removed_count = 0
        ordered_pages: list[list[PdfLine]] = []
        for page_lines in pages:
            filtered_lines = []
            for line in page_lines:
                if _normalize_repeated_line(line.text) in repeated_header_footer:
                    removed_count += 1
                    continue
                filtered_lines.append(line)
            ordered_pages.append(self._order_lines(filtered_lines))

        elements = self._build_elements(ordered_pages)
        text = "\n".join(element.content for element in elements if element.content).strip()
        table_count = sum(1 for element in elements if element.type == "table")

        return ParseResult(
            text=text,
            page_count=page_count,
            metadata={
                "suffix": file_path.suffix.lower(),
                "page_count": page_count,
                "removed_repeated_header_footer_count": removed_count,
            },
            elements=elements,
            parse_quality=ParseQuality(
                page_count=page_count,
                element_count=len(elements),
                table_count=table_count,
                removed_repeated_header_footer_count=removed_count,
            ),
        )

    def _extract_page_lines(self, page, page_number: int) -> list[PdfLine]:
        raw = page.get_text("dict")
        lines: list[PdfLine] = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(str(span.get("text") or "") for span in spans).strip()
                if not text:
                    continue
                font_sizes = [
                    float(span.get("size") or 0)
                    for span in spans
                    if isinstance(span.get("size"), int | float)
                ]
                flags = max(
                    (int(span.get("flags") or 0) for span in spans),
                    default=0,
                )
                lines.append(
                    PdfLine(
                        text=text,
                        page_number=page_number,
                        bbox=[float(item) for item in line.get("bbox", block.get("bbox", []))],
                        font_size=max(font_sizes, default=0.0),
                        flags=flags,
                    )
                )
        return lines

    def _detect_repeated_header_footer(self, pages: list[list[PdfLine]]) -> set[str]:
        if len(pages) < 2:
            return set()
        candidates: dict[str, int] = {}
        for page_lines in pages:
            if not page_lines:
                continue
            page_height = max(
                (line.bbox[3] for line in page_lines if len(line.bbox) >= 4),
                default=0,
            )
            boundary_lines = [
                line
                for line in page_lines
                if len(line.bbox) >= 4
                and (line.bbox[1] <= page_height * 0.12 or line.bbox[3] >= page_height * 0.88)
            ]
            seen_on_page = {_normalize_repeated_line(line.text) for line in boundary_lines}
            for value in seen_on_page:
                if value:
                    candidates[value] = candidates.get(value, 0) + 1
        threshold = max(2, len(pages) // 2 + 1)
        return {value for value, count in candidates.items() if count >= threshold}

    def _order_lines(self, lines: list[PdfLine]) -> list[PdfLine]:
        if len(lines) < 4:
            return sorted(lines, key=lambda line: (_y0(line), _x0(line)))
        page_width = max((_x1(line) for line in lines), default=0)
        centers = sorted((_x0(line) + _x1(line)) / 2 for line in lines)
        gaps = [
            (centers[index + 1] - centers[index], centers[index], centers[index + 1])
            for index in range(len(centers) - 1)
        ]
        large_gaps = [gap for gap in gaps if gap[0] >= page_width * 0.18]
        if not large_gaps:
            return sorted(lines, key=lambda line: (_y0(line), _x0(line)))

        _, left_edge, right_edge = max(large_gaps, key=lambda gap: gap[0])
        split_at = (left_edge + right_edge) / 2
        left = [line for line in lines if (_x0(line) + _x1(line)) / 2 <= split_at]
        right = [line for line in lines if (_x0(line) + _x1(line)) / 2 > split_at]
        if len(left) < 2 or len(right) < 2:
            return sorted(lines, key=lambda line: (_y0(line), _x0(line)))
        return [
            *sorted(left, key=lambda line: (_y0(line), _x0(line))),
            *sorted(right, key=lambda line: (_y0(line), _x0(line))),
        ]

    def _build_elements(self, pages: list[list[PdfLine]]) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        section_path: list[str] = []
        previous_table: TableStructure | None = None
        previous_table_id: str | None = None
        for page_lines in pages:
            table_buffer: list[PdfLine] = []
            pending_title: PdfLine | None = None
            for line in page_lines:
                if _is_table_title(line.text):
                    previous_table, previous_table_id = self._flush_table(
                        table_buffer,
                        elements,
                        section_path,
                        previous_table,
                        previous_table_id,
                    )
                    table_buffer = []
                    pending_title = line
                    continue
                if _is_table_line(line.text):
                    table_buffer.append(line)
                    continue

                previous_table, previous_table_id = self._flush_table(
                    table_buffer,
                    elements,
                    section_path,
                    previous_table,
                    previous_table_id,
                    title_line=pending_title,
                )
                table_buffer = []
                pending_title = None
                element_type = _classify_pdf_line(line)
                if element_type == "heading":
                    section_path = [line.text]
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=line.text,
                        page_start=line.page_number,
                        page_end=line.page_number,
                        section_path=list(section_path),
                        bbox=line.bbox,
                        metadata={
                            "font_size": line.font_size,
                            "font_flags": line.flags,
                        },
                    )
                )
            previous_table, previous_table_id = self._flush_table(
                table_buffer,
                elements,
                section_path,
                previous_table,
                previous_table_id,
                title_line=pending_title,
            )
        return elements

    def _flush_table(
        self,
        table_lines: list[PdfLine],
        elements: list[DocumentElement],
        section_path: list[str],
        previous_table: TableStructure | None,
        previous_table_id: str | None,
        title_line: PdfLine | None = None,
    ) -> tuple[TableStructure | None, str | None]:
        if len(table_lines) < 2:
            return previous_table, previous_table_id
        table = _parse_table_lines(table_lines, title_line.text if title_line else None)
        if table is None:
            return previous_table, previous_table_id
        if previous_table is not None and _is_continuation(previous_table, table):
            table.continuation_of = previous_table_id
        table_id = f"table_{len(elements) + 1}"
        elements.append(
            DocumentElement(
                type="table",
                content=_table_to_text(table),
                page_start=table_lines[0].page_number,
                page_end=table_lines[-1].page_number,
                section_path=list(section_path),
                bbox=_merge_bbox([line.bbox for line in table_lines]),
                metadata={"table_id": table_id, "table": table.model_dump()},
            )
        )
        return table, table_id


def _normalize_repeated_line(text: str) -> str:
    normalized = re.sub(r"\d+", "#", text.strip().lower())
    return re.sub(r"\s+", "", normalized)


def _x0(line: PdfLine) -> float:
    return line.bbox[0] if len(line.bbox) >= 4 else 0.0


def _y0(line: PdfLine) -> float:
    return line.bbox[1] if len(line.bbox) >= 4 else 0.0


def _x1(line: PdfLine) -> float:
    return line.bbox[2] if len(line.bbox) >= 4 else 0.0


def _is_table_title(text: str) -> bool:
    return bool(re.match(r"^\s*(表|Table)\s*[\d一二三四五六七八九十A-Za-z.-]*", text))


def _is_table_line(text: str) -> bool:
    stripped = text.strip()
    if "|" in stripped and stripped.count("|") >= 2:
        return True
    cells = _split_table_cells(stripped)
    return len(cells) >= 2 and any(re.search(r"\d", cell) for cell in cells[1:])


def _split_table_cells(text: str) -> list[str]:
    if "|" in text:
        return [cell.strip() for cell in text.strip("|").split("|") if cell.strip()]
    return [cell.strip() for cell in re.split(r"\s{2,}|\t+", text) if cell.strip()]


def _parse_table_lines(table_lines: list[PdfLine], title: str | None) -> TableStructure | None:
    rows = [_split_table_cells(line.text) for line in table_lines]
    rows = [row for row in rows if row]
    if len(rows) < 2:
        return None
    headers = rows[0]
    return TableStructure(
        title=title,
        headers=headers,
        rows=rows[1:],
        units=_extract_units(" ".join([title or "", *headers])),
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
    unit_match = re.search(r"(单位|Unit)[:：]\s*([^\s，,；;）)]+)", text, re.IGNORECASE)
    if unit_match:
        return unit_match.group(2)
    bracket_match = re.search(r"[（(]([%％\u4e00-\u9fff]{1,8})[）)]", text)
    return bracket_match.group(1) if bracket_match else None


def _is_continuation(previous: TableStructure, current: TableStructure) -> bool:
    if previous.headers and current.headers and previous.headers == current.headers:
        return True
    if previous.headers and current.rows:
        return len(previous.headers) == len(current.rows[0])
    return False


def _classify_pdf_line(line: PdfLine) -> str:
    text = line.text.strip()
    if re.match(r"^(#{1,6}\s+|第[一二三四五六七八九十百千万\d]+[章节]\b)", text):
        return "heading"
    if re.match(r"^([-*•]\s+|\d+[.)、]\s+|[一二三四五六七八九十]+[、.]\s*)", text):
        return "list"
    if line.font_size >= 14 and len(text) <= 80:
        return "heading"
    return "paragraph"


def _merge_bbox(bboxes: list[list[float]]) -> list[float] | None:
    valid = [bbox for bbox in bboxes if len(bbox) >= 4]
    if not valid:
        return None
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]
