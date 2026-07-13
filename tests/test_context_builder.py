from backend.app.context import BasicContextBuilder, ContextBuildRequest
from backend.app.rerankers import RerankedChunk


def test_context_builder_includes_document_structure_metadata():
    chunk = RerankedChunk(
        id="law-12",
        original_score=0.9,
        rerank_score=0.95,
        text="第十二条 劳动者就业，不因民族、种族、性别、宗教信仰不同而受歧视。",
        document_id=12,
        knowledge_base_id=1,
        chunk_index=3,
        metadata={
            "source": "中国劳动法.pdf",
            "section_path": ["中华人民共和国劳动法", "第二章 促进就业"],
            "chapter_label": "第二章",
            "chapter_title": "促进就业",
            "article_label": "第十二条",
        },
    )

    result = BasicContextBuilder().build(
        ContextBuildRequest(
            query="劳动法第二章讲的是什么？",
            chunks=[chunk],
        )
    )

    assert "Document: 中国劳动法.pdf" in result.context_text
    assert "Section Path: 中华人民共和国劳动法 > 第二章 促进就业" in result.context_text
    assert "Chapter: 第二章" in result.context_text
    assert "Chapter Title: 促进就业" in result.context_text
    assert "Article: 第十二条" in result.context_text
    assert "正文:\n第十二条" in result.context_text
