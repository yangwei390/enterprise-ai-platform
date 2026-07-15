from backend.app.cleaners import CleanResult
from backend.app.config.settings import settings
from backend.app.parsers import DocumentElement, ParseResult, TableStructure
from backend.app.pipeline.base import PipelineContext
from backend.app.pipeline.document_pipeline import ChunkStep


def test_parse_result_elements_enter_chunk_main_chain():
    context = _pipeline_context(
        text="Chapter One\nParagraph one.",
        elements=[
            _element("heading", "Chapter One", section_path=["Chapter One"], page_start=1),
            _element("paragraph", "Paragraph one.", section_path=["Chapter One"], page_start=1),
        ],
    )

    result = ChunkStep().run(context)

    assert result.chunk_result is not None
    assert result.chunk_result.strategy == "parent_child"
    assert result.metadata["chunking"]["actual_strategy"] == "parent_child"
    assert result.chunk_result.chunks[0].metadata["structure_source"] == "parser_elements"


def test_elements_are_not_lost_after_cleaner_text_processing():
    context = _pipeline_context(
        text="Cleaned heading\nCleaned body",
        elements=[
            _element("heading", "Raw Heading", section_path=["Raw Heading"]),
            _element("paragraph", "Raw body", section_path=["Raw Heading"]),
        ],
    )

    result = ChunkStep().run(context)

    assert result.chunk_result is not None
    assert any(
        chunk.metadata.get("heading_title") == "Raw Heading"
        for chunk in result.chunk_result.chunks
    )


def test_heading_boundary_and_paragraph_ownership():
    context = _pipeline_context(
        text="H1\nP1\nH2\nP2",
        elements=[
            _element("heading", "H1", section_path=["H1"], page_start=1),
            _element("paragraph", "P1", section_path=["H1"], page_start=1),
            _element("heading", "H2", section_path=["H2"], page_start=2),
            _element("paragraph", "P2", section_path=["H2"], page_start=2),
        ],
    )

    result = ChunkStep().run(context)
    chunks = result.chunk_result.chunks

    assert any("H1" in chunk.text and "P1" in chunk.text for chunk in chunks)
    assert any("H2" in chunk.text and "P2" in chunk.text for chunk in chunks)
    assert not any("P1" in chunk.text and "H2" in chunk.text for chunk in chunks)


def test_list_elements_group_with_section_metadata():
    context = _pipeline_context(
        text="Tasks\n- one\n- two",
        elements=[
            _element("heading", "Tasks", section_path=["Tasks"]),
            _element("list", "- one", section_path=["Tasks"]),
            _element("list", "- two", section_path=["Tasks"]),
        ],
    )

    result = ChunkStep().run(context)
    chunk = result.chunk_result.chunks[0]

    assert "list" in chunk.metadata["element_types"]
    assert chunk.metadata["section_path"] == ["Tasks"]


def test_table_gets_independent_chunk_with_metadata():
    context = _pipeline_context(
        text="Sales\nTable 1\nItem | Value\nRevenue | 100",
        elements=[
            _element("heading", "Sales", section_path=["Sales"]),
            _table_element(
                title="Table 1",
                headers=["Item", "Value"],
                rows=[["Revenue", "100"]],
                units="USD",
                section_path=["Sales"],
                page_start=3,
            ),
        ],
    )

    result = ChunkStep().run(context)
    table_chunk = next(
        chunk
        for chunk in result.chunk_result.chunks
        if "table" in chunk.metadata["element_types"]
    )

    assert table_chunk.metadata["table_title"] == "Table 1"
    assert table_chunk.metadata["table_headers"] == ["Item", "Value"]
    assert table_chunk.metadata["table_units"] == "USD"
    assert table_chunk.metadata["page_start"] == 3
    assert "Item | Value" in table_chunk.text
    assert "Revenue | 100" in table_chunk.text


def test_large_table_splits_by_row_groups_and_repeats_headers(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_CHILD_MAX_CHARS", 45)
    rows = [[f"row{i}", "100"] for i in range(6)]
    context = _pipeline_context(
        text="Big Table",
        elements=[
            _table_element(
                title="Table Big",
                headers=["Name", "Value"],
                rows=rows,
                units="kg",
            )
        ],
    )

    result = ChunkStep().run(context)
    table_chunks = [
        chunk for chunk in result.chunk_result.chunks if "table" in chunk.metadata["element_types"]
    ]

    assert len(table_chunks) > 1
    assert all("Name | Value" in chunk.text for chunk in table_chunks)
    assert all(chunk.metadata["table_units"] == "kg" for chunk in table_chunks)


def test_continuation_table_metadata_is_preserved():
    context = _pipeline_context(
        text="Table continuation",
        elements=[
            _table_element(
                title="Table 1",
                headers=["Name", "Value"],
                rows=[["row1", "1"]],
                table_id="table_1",
            ),
            _table_element(
                headers=["Name", "Value"],
                rows=[["row2", "2"]],
                continuation_of="table_1",
                page_start=2,
            ),
        ],
    )

    result = ChunkStep().run(context)
    continuation = [
        chunk
        for chunk in result.chunk_result.chunks
        if chunk.metadata.get("table_continuation") == "table_1"
    ]

    assert continuation
    assert continuation[0].metadata["page_start"] == 2


def test_parent_child_relationship_uses_heading_scope(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_EMBED_PARENT", True)
    context = _pipeline_context(
        text="Scope\nBody",
        elements=[
            _element("heading", "Scope", section_path=["Scope"]),
            _element("paragraph", "Body", section_path=["Scope"]),
        ],
    )

    result = ChunkStep().run(context)
    parent = next(
        chunk
        for chunk in result.chunk_result.chunks
        if chunk.metadata["chunk_role"] == "parent"
    )
    child = next(
        chunk
        for chunk in result.chunk_result.chunks
        if chunk.metadata["chunk_role"] == "child"
    )

    assert parent.metadata["section_path"] == ["Scope"]
    assert child.metadata["parent_chunk_id"] == parent.metadata["chunk_uid"]
    assert child.metadata["chunk_uid"] in parent.metadata["child_chunk_ids"]


def test_empty_elements_fallback_to_existing_structure_chain():
    context = _pipeline_context("Plain paragraph only.", elements=[])

    result = ChunkStep().run(context)

    assert result.chunk_result is not None
    assert result.chunk_result.strategy in {"recursive", "fixed"}
    assert not any(
        chunk.metadata.get("structure_source") == "parser_elements"
        for chunk in result.chunk_result.chunks
    )


def test_legal_and_markdown_without_elements_do_not_regress():
    legal = _pipeline_context(
        "中华人民共和国示例法\n\n第一章 总则\n\n第一条 示例内容。",
        elements=[],
        filename="law.txt",
    )
    markdown = _pipeline_context("# Title\n\nBody", elements=[], filename="doc.md")

    legal_result = ChunkStep().run(legal)
    markdown_result = ChunkStep().run(markdown)

    assert legal_result.chunk_result.strategy == "legal_structure"
    assert markdown_result.chunk_result.strategy == "markdown"


def test_docx_parser_elements_are_chunked_without_fixed_fallback():
    context = _pipeline_context(
        "Docx Heading\nDocx body",
        elements=[
            _element("heading", "Docx Heading", section_path=["Docx Heading"]),
            _element("paragraph", "Docx body", section_path=["Docx Heading"]),
        ],
        filename="doc.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    result = ChunkStep().run(context)

    assert result.chunk_result.strategy == "parent_child"
    assert result.chunk_result.chunks[0].metadata["heading_title"] == "Docx Heading"


def _pipeline_context(
    text: str,
    *,
    elements: list[DocumentElement],
    filename: str = "example.pdf",
    mime_type: str = "application/pdf",
) -> PipelineContext:
    class Document:
        id = 9
        knowledge_base_id = 4

    document = Document()
    document.filename = filename
    document.original_filename = filename
    document.mime_type = mime_type

    return PipelineContext(
        document=document,
        parse_result=ParseResult(
            text=text,
            page_count=2,
            metadata={},
            elements=elements,
        ),
        clean_result=CleanResult(
            text=text.strip(),
            original_length=len(text),
            cleaned_length=len(text.strip()),
            metadata={},
        ),
        metadata={
            "suffix": f".{filename.rsplit('.', 1)[-1]}",
            "parser": "PdfParser",
            "cleaner": "BasicTextCleaner",
        },
    )


def _element(
    element_type: str,
    content: str,
    *,
    section_path: list[str] | None = None,
    page_start: int | None = None,
) -> DocumentElement:
    return DocumentElement(
        type=element_type,
        content=content,
        page_start=page_start,
        page_end=page_start,
        section_path=section_path or [],
    )


def _table_element(
    *,
    headers: list[str],
    rows: list[list[str]],
    title: str | None = None,
    units: str | None = None,
    continuation_of: str | None = None,
    table_id: str | None = None,
    section_path: list[str] | None = None,
    page_start: int | None = None,
) -> DocumentElement:
    table = TableStructure(
        title=title,
        headers=headers,
        rows=rows,
        units=units,
        continuation_of=continuation_of,
    )
    return DocumentElement(
        type="table",
        content="\n".join([" | ".join(headers), *(" | ".join(row) for row in rows)]),
        page_start=page_start,
        page_end=page_start,
        section_path=section_path or [],
        metadata={"table_id": table_id, "table": table.model_dump()},
    )
