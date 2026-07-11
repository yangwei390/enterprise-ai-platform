import json

from backend.app.api.debug import router as debug_router
from backend.app.chunkers import (
    Chunk,
    ChunkerFactory,
    FixedChunker,
    LegalStructureChunker,
    MarkdownChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
)
from backend.app.chunkers.router import ChunkStrategyRouter
from backend.app.cleaners import CleanResult
from backend.app.config.settings import settings
from backend.app.documents import (
    ChunkMetadataBuilder,
    DocumentClassifier,
    StructureQueryHintParser,
)
from backend.app.documents.structure import parse_chinese_number
from backend.app.documents.structures import (
    LegalStructureParser,
    MarkdownStructureParser,
)
from backend.app.embeddings import EmbeddingFactory
from backend.app.parsers import ParseResult
from backend.app.pipeline.base import PipelineContext
from backend.app.pipeline.document_pipeline import ChunkStep
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps import NeighborExpansionStep, SoftBoostStep
from evaluation.v2.metrics.chunking import (
    ArticleCoverageMetric,
    StructurePathCoverageMetric,
)
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult
from fastapi import FastAPI
from fastapi.testclient import TestClient

LEGAL_TEXT = """中华人民共和国示例法

第一章 总则

第一条 为了规范示例事项，制定本法。
本条第二款继续说明立法目的。

第二条 本法适用于示例活动。

第二章 促进事项

第十条 国家促进示例事项发展。

第十一条 地方人民政府应当支持示例事项。

第十二条 有关部门应当完善服务。

第三章 附则

第二十条 本法自公布之日起施行。
"""


MARKDOWN_TEXT = """# RAG

Intro

## Retriever

Retriever text.

```python
# Not A Heading
```

### BM25

BM25 text.
"""


def _metadata() -> dict:
    return {
        "document_id": 9,
        "knowledge_base_id": 4,
        "source": "示例法.pdf",
        "parser": "TextParser",
        "cleaner": "BasicTextCleaner",
        "page_count": 1,
    }


def test_document_classifier_detects_legal():
    result = DocumentClassifier().classify(text=LEGAL_TEXT, filename="示例法.txt")
    assert result.document_type == "legal"


def test_document_classifier_detects_markdown():
    result = DocumentClassifier().classify(text=MARKDOWN_TEXT, filename="readme.md")
    assert result.document_type == "markdown"


def test_document_classifier_falls_back_plain_text():
    result = DocumentClassifier().classify(text="普通文本，没有结构")
    assert result.document_type == "plain_text"


def test_chinese_number_parser():
    assert parse_chinese_number("第二十一") == 21
    assert parse_chinese_number("第一百零二") == 102
    assert parse_chinese_number("12") == 12


def test_legal_parser_detects_chapters():
    structure = LegalStructureParser().parse(LEGAL_TEXT, _metadata())
    chapters = [node for node in structure.nodes if node.node_type == "chapter"]
    assert [chapter.metadata["chapter_number"] for chapter in chapters] == [1, 2, 3]


def test_legal_parser_detects_articles():
    structure = LegalStructureParser().parse(LEGAL_TEXT, _metadata())
    articles = [node for node in structure.nodes if node.node_type == "article"]
    assert {article.metadata["article_number"] for article in articles} >= {1, 10, 11}


def test_legal_parser_preserves_article_paragraphs():
    structure = LegalStructureParser().parse(LEGAL_TEXT, _metadata())
    article_one = next(
        node for node in structure.nodes if node.metadata.get("article_number") == 1
    )
    assert "本条第二款继续说明立法目的" in article_one.text


def test_legal_parser_builds_section_path():
    structure = LegalStructureParser().parse(LEGAL_TEXT, _metadata())
    article_ten = next(
        node for node in structure.nodes if node.metadata.get("article_number") == 10
    )
    assert "第二章 促进事项" in article_ten.metadata["section_path"]


def test_legal_parser_fail_open_plain_text():
    structure = LegalStructureParser().parse("无结构文本", _metadata())
    assert structure.document_type == "plain_text"


def test_markdown_parser_builds_heading_tree():
    structure = MarkdownStructureParser().parse(MARKDOWN_TEXT, _metadata())
    headings = [node for node in structure.nodes if node.node_type == "heading"]
    assert [heading.title for heading in headings] == ["RAG", "Retriever", "BM25"]


def test_markdown_parser_ignores_fenced_code_heading():
    structure = MarkdownStructureParser().parse(MARKDOWN_TEXT, _metadata())
    titles = [node.title for node in structure.nodes]
    assert "Not A Heading" not in titles


def test_chunk_strategy_router_selects_legal():
    decision = ChunkStrategyRouter().route(document_type="legal")
    assert decision.actual_strategy == "legal_structure"


def test_chunk_strategy_router_selects_markdown():
    decision = ChunkStrategyRouter().route(document_type="markdown")
    assert decision.actual_strategy == "markdown"


def test_chunk_strategy_router_selects_recursive():
    decision = ChunkStrategyRouter().route(document_type="plain_text")
    assert decision.actual_strategy == "recursive"


def test_fixed_strategy_still_works():
    result = FixedChunker(chunk_size=10, chunk_overlap=2).chunk("0123456789abcdef", _metadata())
    assert result.strategy == "fixed"
    assert result.chunks[0].metadata["chunk_uid"]


def test_recursive_chunker_prefers_sentence_boundary():
    result = RecursiveChunker(chunk_size=8, chunk_overlap=0, min_chars=3).chunk(
        "第一句。第二句。第三句。",
        _metadata(),
    )
    assert result.chunks[0].text.endswith("。")


def test_recursive_chunker_respects_max_size():
    result = RecursiveChunker(chunk_size=20, chunk_overlap=0, min_chars=3).chunk(
        "很长的文本。" * 20,
        _metadata(),
    )
    assert max(len(chunk.text) for chunk in result.chunks) <= 20


def test_recursive_chunker_avoids_tiny_chunks():
    result = RecursiveChunker(chunk_size=20, chunk_overlap=0, min_chars=5).chunk(
        "短。很长很长的一句话。末尾。",
        _metadata(),
    )
    assert all(len(chunk.text) >= 5 for chunk in result.chunks)


def test_legal_chunker_does_not_cross_chapter():
    result = LegalStructureChunker().chunk(LEGAL_TEXT, _metadata())
    second_chapter_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.get("chapter_number") == 2
    ]
    assert second_chapter_chunks
    assert all("第一章" not in chunk.text for chunk in second_chapter_chunks)
    assert all("第三章" not in chunk.text for chunk in second_chapter_chunks)


def test_legal_chunker_preserves_article_range():
    result = LegalStructureChunker().chunk(LEGAL_TEXT, _metadata())
    article_ten = next(chunk for chunk in result.chunks if "第十条" in chunk.text)
    assert article_ten.metadata["article_start"] == 10
    assert article_ten.metadata["article_end"] == 10


def test_long_article_recursive_split(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_LEGAL_MAX_CHARS", 120)
    text = "中华人民共和国示例法\n第一章 总则\n第一条 " + ("很长内容。" * 80)
    result = LegalStructureChunker().chunk(text, _metadata())
    assert len(result.chunks) > 1
    assert all(chunk.metadata["article_start"] == 1 for chunk in result.chunks)


def test_markdown_chunker_preserves_section_path():
    result = MarkdownChunker().chunk(MARKDOWN_TEXT, _metadata())
    bm25 = next(chunk for chunk in result.chunks if "BM25 text" in chunk.text)
    assert bm25.metadata["section_path"][-1] == "BM25"


def test_parent_child_chunk_links():
    result = ParentChildChunker().chunk("父子切片测试。" * 100, _metadata())
    assert all(chunk.metadata.get("parent_chunk_id") for chunk in result.chunks)


def test_parent_child_chunk_ids_stable():
    first = ParentChildChunker().chunk("稳定文本。" * 30, _metadata())
    second = ParentChildChunker().chunk("稳定文本。" * 30, _metadata())
    assert (
        first.chunks[0].metadata["parent_chunk_id"]
        == second.chunks[0].metadata["parent_chunk_id"]
    )


def test_chunk_uid_stable_for_same_content():
    builder = ChunkMetadataBuilder()
    chunk = Chunk(
        document_id=1,
        knowledge_base_id=1,
        chunk_index=0,
        text="same",
        start_offset=0,
        end_offset=4,
    )
    first = builder.build(chunk=chunk, source_metadata=_metadata(), strategy="fixed")
    second = builder.build(chunk=chunk, source_metadata=_metadata(), strategy="fixed")
    assert first["chunk_uid"] == second["chunk_uid"]


def test_chunk_metadata_json_serializable():
    result = LegalStructureChunker().chunk(LEGAL_TEXT, _metadata())
    json.dumps(result.chunks[0].metadata, ensure_ascii=False)


def test_semantic_chunker_uses_embedding_provider(monkeypatch):
    class FakeEmbedding:
        def embed_text(self, text):
            return [1.0, 0.0] if "A" in text else [0.0, 1.0]

    monkeypatch.setattr(settings, "CHUNK_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(EmbeddingFactory, "get_embedding", lambda: FakeEmbedding())
    result = SemanticChunker().chunk("A 段落\n\nB 段落", _metadata())
    assert result.strategy == "semantic"


def test_semantic_chunker_fail_open_recursive(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(
        EmbeddingFactory,
        "get_embedding",
        lambda: (_ for _ in ()).throw(RuntimeError("no embedding")),
    )
    result = SemanticChunker().chunk("A 段落\n\nB 段落", _metadata())
    assert result.strategy == "recursive"
    assert result.metadata["fallback_used"] is True


def test_structure_query_hint_parses_chapter():
    hint = StructureQueryHintParser().parse("劳动法第二章说什么")
    assert hint.chapter_number == 2


def test_structure_query_hint_parses_article():
    hint = StructureQueryHintParser().parse("第十条规定是什么")
    assert hint.article_number == 10


def test_structure_soft_boost_metadata():
    chunk = RetrievedChunk(
        id="1",
        score=1.0,
        text="第二章",
        document_id=1,
        knowledge_base_id=1,
        chunk_index=0,
        metadata={"chapter_number": 2},
    )
    context = RetrieverPipelineContext(query="第二章", top_k=1, fused_chunks=[chunk])
    result = SoftBoostStep().run(context)
    assert result.metadata["structure_soft_boost_applied"] is True


def test_neighbor_expansion_stays_in_parent():
    parent = _reranked(2, {"parent_chunk_id": "p1"})
    neighbor = _reranked(1, {"parent_chunk_id": "p2"})
    assert NeighborExpansionStep(_lookup({1: neighbor})).run(
        RetrieverPipelineContext(query="q", top_k=5, reranked_chunks=[parent])
    ).metadata["neighbor_expansion"]["added_chunk_count"] == 0


def test_neighbor_expansion_stays_in_chapter():
    parent = _reranked(2, {"chapter_number": 2})
    neighbor = _reranked(1, {"chapter_number": 3})
    result = NeighborExpansionStep(_lookup({1: neighbor})).run(
        RetrieverPipelineContext(query="q", top_k=5, reranked_chunks=[parent])
    )
    assert result.metadata["neighbor_expansion"]["added_chunk_count"] == 0


def test_neighbor_expansion_old_chunk_compat():
    parent = _reranked(2, {})
    neighbor = _reranked(1, {})
    result = NeighborExpansionStep(_lookup({1: neighbor})).run(
        RetrieverPipelineContext(query="q", top_k=5, reranked_chunks=[parent])
    )
    assert result.metadata["neighbor_expansion"]["added_chunk_count"] == 1


def test_ingestion_pipeline_auto_strategy():
    context = _pipeline_context(LEGAL_TEXT)
    result = ChunkStep().run(context)
    assert result.chunk_result is not None
    assert result.chunk_result.strategy == "legal_structure"


def test_ingestion_pipeline_force_fixed_strategy(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_STRATEGY", "fixed")
    context = _pipeline_context(LEGAL_TEXT)
    result = ChunkStep().run(context)
    assert result.chunk_result is not None
    assert result.chunk_result.strategy == "fixed"


def test_ingestion_fallback_chain(monkeypatch):
    monkeypatch.setattr(
        "backend.app.pipeline.document_pipeline.DocumentClassifier.classify",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    context = _pipeline_context(LEGAL_TEXT)
    result = ChunkStep().run(context)
    assert result.chunk_result is not None
    assert result.chunk_result.strategy == "fixed"


def test_debug_document_structure_api(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.debug._build_document_chunk_preview",
        lambda document_id, db, strategy: _debug_preview(),
    )
    client = _debug_client()
    response = client.get("/debug/documents/1/structure")
    assert response.status_code == 200
    assert response.json()["data"]["document_type"] == "legal"


def test_debug_document_chunks_api(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.debug._build_document_chunk_preview",
        lambda document_id, db, strategy: _debug_preview(),
    )
    client = _debug_client()
    response = client.get("/debug/documents/1/chunks")
    assert response.status_code == 200
    assert response.json()["data"]["chunks"][0]["chunk_uid"] == "u1"


def test_chunk_quality_metrics():
    case = EvaluationCase(
        id="chunk",
        target="rag",
        expected={"articles": [10]},
        metrics=["structure_path_coverage", "article_coverage"],
    )
    target_result = EvaluationTargetResult(
        target="rag",
        chunks=[
            {
                "text": "第十条",
                "metadata": {
                    "section_path": ["法", "第二章"],
                    "article_start": 10,
                    "article_end": 10,
                    "chunk_role": "child",
                    "parent_chunk_id": "p1",
                },
            }
        ],
    )
    coverage = _run_metric(StructurePathCoverageMetric(), case, target_result)
    article = _run_metric(ArticleCoverageMetric(), case, target_result)
    assert coverage.value == 1
    assert article.value == 1


def test_existing_rag_pipeline_still_works():
    assert ChunkerFactory.get_chunker(strategy="fixed").chunk("hello", _metadata()).chunks


def test_chat_agent_workflow_apis_still_exist():
    from backend.app.api.agent import router as agent_router
    from backend.app.api.chat import router as chat_router
    from backend.app.api.workflow import router as workflow_router

    assert agent_router.routes
    assert chat_router.routes
    assert workflow_router.routes


def test_real_legal_chunking_smoke():
    result = LegalStructureChunker().chunk(LEGAL_TEXT, _metadata())
    assert result.chunks
    assert any(chunk.metadata.get("chapter_number") == 2 for chunk in result.chunks)
    assert all(chunk.metadata.get("chunk_uid") for chunk in result.chunks)


def test_real_markdown_chunking_smoke():
    result = MarkdownChunker().chunk(MARKDOWN_TEXT, _metadata())
    assert any(chunk.metadata.get("heading_title") == "BM25" for chunk in result.chunks)


def test_real_ingestion_chunking_smoke():
    context = _pipeline_context(MARKDOWN_TEXT)
    result = ChunkStep().run(context)
    assert result.chunk_result is not None
    assert result.chunk_result.total_chunks > 0
    assert result.chunk_result.chunks[0].metadata.get("chunk_uid")


def _pipeline_context(text: str) -> PipelineContext:
    class Document:
        id = 9
        knowledge_base_id = 4
        filename = "example.txt"
        original_filename = "example.txt"
        mime_type = "text/plain"

    return PipelineContext(
        document=Document(),
        parse_result=ParseResult(text=text, page_count=1, metadata={}),
        clean_result=CleanResult(
            text=text,
            original_length=len(text),
            cleaned_length=len(text),
            metadata={},
        ),
        metadata={"suffix": ".txt", "parser": "TextParser", "cleaner": "BasicTextCleaner"},
    )


def _lookup(items):
    class Lookup:
        def find_neighbor(self, *, document_id, knowledge_base_id, chunk_index):
            return items.get(chunk_index)

    return Lookup()


def _reranked(index: int, metadata: dict) -> RerankedChunk:
    return RerankedChunk(
        id=str(index),
        original_score=1.0,
        rerank_score=1.0,
        text=f"chunk {index}",
        document_id=1,
        knowledge_base_id=1,
        chunk_index=index,
        metadata=metadata,
    )


def _debug_client() -> TestClient:
    app = FastAPI()
    app.include_router(debug_router)
    return TestClient(app)


def _debug_preview() -> dict:
    return {
        "structure": {
            "document_type": "legal",
            "metadata": {"node_count": 1, "max_depth": 1},
            "nodes": [
                {
                    "id": "root",
                    "node_type": "document",
                    "title": "示例法",
                    "level": 0,
                    "path": ["示例法"],
                    "metadata": {},
                }
            ],
        },
        "chunk_result": {
            "strategy": "legal_structure",
            "total_chunks": 1,
            "metadata": {},
        },
        "chunks": [
            {
                "chunk_index": 0,
                "chunk_uid": "u1",
                "text_preview": "第十条",
                "chunk_role": "child",
                "parent_chunk_id": "p1",
                "section_path": ["示例法"],
                "metadata": {},
            }
        ],
    }


def _run_metric(metric, case, result):
    import asyncio

    return asyncio.run(metric.evaluate(case, result, None))
