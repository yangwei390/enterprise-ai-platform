from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
from backend.app.parsers import ParseResult, ParserFactory
from backend.app.parsers.docx_parser import DocxParser
from backend.app.parsers.pdf_parser import PdfParser
from backend.app.parsers.text_parser import TextParser


def test_text_parser_keeps_text_and_adds_elements(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("标题\n第一段内容。\n- 列表项\n", encoding="utf-8")

    result = TextParser().parse(file_path)

    assert result.text == "标题\n第一段内容。\n- 列表项\n"
    assert result.metadata["suffix"] == ".txt"
    assert [element.type for element in result.elements] == ["heading", "paragraph", "list"]


def test_pdf_parser_keeps_plain_pdf_text(tmp_path: Path):
    file_path = tmp_path / "plain.pdf"
    _write_pdf(file_path, [["Plain PDF Title", "Plain paragraph content."]])

    result = PdfParser().parse(file_path)

    assert "Plain PDF Title" in result.text
    assert "Plain paragraph content" in result.text
    assert result.page_count == 1
    assert result.metadata["suffix"] == ".pdf"
    assert result.parse_quality is not None
    assert result.parse_quality.element_count >= 2


def test_docx_parser_keeps_text_and_table(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    _write_docx(
        file_path,
        paragraphs=[("Heading1", "年度报告"), (None, "正文段落。")],
        tables=[[["项目", "金额 单位：万元"], ["收入", "100"]]],
    )

    result = DocxParser().parse(file_path)

    assert "年度报告" in result.text
    assert "正文段落" in result.text
    table = next(element for element in result.elements if element.type == "table")
    assert table.metadata["table"]["headers"] == ["项目", "金额 单位：万元"]
    assert table.metadata["table"]["rows"] == [["收入", "100"]]
    assert table.metadata["table"]["units"] == "万元"


def test_parser_factory_supports_docx(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    _write_docx(file_path, paragraphs=[(None, "docx text")])

    parser = ParserFactory.get_parser(file_path)

    assert isinstance(parser, DocxParser)


def test_pdf_parser_orders_two_columns(tmp_path: Path):
    file_path = tmp_path / "columns.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    page.insert_text((40, 40), "Left column first")
    page.insert_text((40, 80), "Left column second")
    page.insert_text((230, 40), "Right column first")
    page.insert_text((230, 80), "Right column second")
    document.save(file_path)
    document.close()

    result = PdfParser().parse(file_path)

    assert result.text.index("Left column first") < result.text.index("Left column second")
    assert result.text.index("Left column second") < result.text.index("Right column first")
    assert result.text.index("Right column first") < result.text.index("Right column second")


def test_pdf_parser_extracts_table_headers_rows_units(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    _write_pdf(
        file_path,
        [
            [
                "Sales Data",
                "Table 1 Sales Unit: USD",
                "Item  2023  2024",
                "Revenue  100  120",
                "Profit  20  30",
            ]
        ],
    )

    result = PdfParser().parse(file_path)
    table = next(element for element in result.elements if element.type == "table")

    assert table.metadata["table"]["title"] == "Table 1 Sales Unit: USD"
    assert table.metadata["table"]["headers"] == ["Item", "2023", "2024"]
    assert table.metadata["table"]["rows"] == [
        ["Revenue", "100", "120"],
        ["Profit", "20", "30"],
    ]
    assert table.metadata["table"]["units"] == "USD"


def test_pdf_parser_removes_repeated_headers_and_footers(tmp_path: Path):
    file_path = tmp_path / "header-footer.pdf"
    _write_pdf(
        file_path,
            [
                ["Company Report", "First page body", "Page 1"],
                ["Company Report", "Second page body", "Page 2"],
            ],
        y_positions=[20, 100, 280],
    )

    result = PdfParser().parse(file_path)

    assert "First page body" in result.text
    assert "Second page body" in result.text
    assert "Company Report" not in result.text
    assert "Page 1" not in result.text
    assert result.parse_quality is not None
    assert result.parse_quality.removed_repeated_header_footer_count >= 2


def test_pdf_parser_marks_cross_page_table_continuation(tmp_path: Path):
    file_path = tmp_path / "continued-table.pdf"
    _write_pdf(
        file_path,
            [
                ["Table 1 Sales Unit: USD", "Item  2023  2024", "Revenue  100  120"],
                ["Item  2023  2024", "Profit  20  30"],
            ],
    )

    result = PdfParser().parse(file_path)
    tables = [element for element in result.elements if element.type == "table"]

    assert len(tables) == 2
    first_id = tables[0].metadata["table_id"]
    assert tables[1].metadata["table"]["continuation_of"] == first_id


def test_old_parse_result_shape_remains_compatible():
    result = ParseResult(text="legacy", metadata={"source": "old"})

    assert result.text == "legacy"
    assert result.metadata["source"] == "old"
    assert result.elements == []
    assert result.parse_quality is None


def _write_pdf(
    file_path: Path,
    pages: list[list[str]],
    y_positions: list[int] | None = None,
) -> None:
    document = fitz.open()
    for lines in pages:
        page = document.new_page(width=400, height=300)
        positions = y_positions or [40 + index * 24 for index in range(len(lines))]
        for line, y in zip(lines, positions, strict=True):
            page.insert_text((40, y), line)
    document.save(file_path)
    document.close()


def _write_docx(
    file_path: Path,
    paragraphs: list[tuple[str | None, str]],
    tables: list[list[list[str]]] | None = None,
) -> None:
    body_parts = [_docx_paragraph(text, style) for style, text in paragraphs]
    for table in tables or []:
        body_parts.append(_docx_table(table))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}</w:body></w:document>"
    )
    with ZipFile(file_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)


def _docx_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{text}</w:t></w:r></w:p>"


def _docx_table(rows: list[list[str]]) -> str:
    row_xml = []
    for row in rows:
        cells = "".join(f"<w:tc>{_docx_paragraph(cell)}</w:tc>" for cell in row)
        row_xml.append(f"<w:tr>{cells}</w:tr>")
    return f"<w:tbl>{''.join(row_xml)}</w:tbl>"
