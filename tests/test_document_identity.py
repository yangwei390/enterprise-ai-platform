from types import SimpleNamespace

from backend.app.cleaners import CleanResult
from backend.app.exceptions import BusinessException
from backend.app.parsers import DocumentElement, ParseResult
from backend.app.pipeline.base import PipelineContext
from backend.app.pipeline.document_identity import DocumentIdentityAnalyzer
from backend.app.pipeline.document_pipeline import DocumentIdentityStep
from backend.app.retrievers.metadata_filter import AutoMetadataFilterBuilder
from backend.app.services.document import DocumentService
from pytest import raises


def test_random_filename_does_not_override_heading_title():
    result = _identity(
        filename="a8f3c9.pdf",
        text="中华人民共和国劳动法\n第一章 总则\n本文规定劳动关系。",
        elements=[_heading("中华人民共和国劳动法")],
    )

    assert result.document_title == "中华人民共和国劳动法"
    assert result.identity_source == "element_heading"


def test_title_extraction_from_parser_heading():
    result = _identity(
        filename="random.docx",
        text="员工手册\n薪资制度\n考勤制度",
        elements=[_heading("员工手册")],
    )

    assert result.document_title == "员工手册"
    assert "员工手册" in result.summary


def test_alias_retrieval_prefers_document_identity():
    document = _document(
        1,
        filename="a8f3c9.pdf",
        identity={
            "document_title": "中华人民共和国劳动法",
            "aliases": ["劳动法", "Labour Law"],
            "keywords": [],
            "entities": {},
            "summary": "法律文本",
        },
    )

    result = _builder([document]).build("劳动法第二章", knowledge_base_id=4)

    assert result.candidate_document_ids == [1]
    assert result.source_hints == ["中华人民共和国劳动法"]


def test_keyword_retrieval_uses_identity_keywords():
    document = _document(
        2,
        filename="random.pdf",
        identity={
            "document_title": "员工手册",
            "aliases": [],
            "keywords": ["薪资制度", "绩效"],
            "entities": {},
            "summary": "员工制度说明",
        },
    )

    result = _builder([document]).build("薪资制度怎么规定", knowledge_base_id=4)

    assert result.candidate_document_ids == [2]


def test_summary_retrieval_uses_identity_summary():
    document = _document(
        3,
        filename="random.pdf",
        identity={
            "document_title": "安装指南",
            "aliases": [],
            "keywords": [],
            "entities": {},
            "summary": "介绍平台部署安装流程和环境准备。",
        },
    )

    result = _builder([document]).build("环境准备怎么做", knowledge_base_id=4)

    assert result.candidate_document_ids == [3]


def test_entity_retrieval_uses_identity_entities():
    document = _document(
        4,
        filename="random.pdf",
        identity={
            "document_title": "组织制度",
            "aliases": [],
            "keywords": [],
            "entities": {"organizations": ["人力资源部"], "laws": []},
            "summary": "制度说明",
        },
    )

    result = _builder([document]).build("人力资源部职责", knowledge_base_id=4)

    assert result.candidate_document_ids == [4]


def test_fallback_to_filename_when_identity_missing():
    document = _document(5, filename="fallback-manual.pdf", identity=None)

    result = _builder([document]).build("fallback manual", knowledge_base_id=4)

    assert result.candidate_document_ids == [5]


def test_document_identity_step_writes_context_metadata():
    class Document:
        filename = "random.pdf"
        original_filename = "random.pdf"

    context = PipelineContext(
        document=Document(),
        parse_result=ParseResult(
            text="中华人民共和国劳动法\n第一章 总则",
            elements=[_heading("中华人民共和国劳动法")],
            metadata={},
        ),
        clean_result=CleanResult(
            text="中华人民共和国劳动法\n第一章 总则",
            original_length=13,
            cleaned_length=13,
            metadata={},
        ),
    )

    result = DocumentIdentityStep().run(context)

    identity = result.metadata["document_identity"]
    assert identity["document_title"] == "中华人民共和国劳动法"
    assert identity["category"] == "legal"
    assert identity["llm_used"] is False


def test_identity_persisted_into_document_metadata():
    document = _document(6, filename="random.pdf", identity=None)
    document.document_metadata = {"existing": {"keep": True}}
    repository = FakeDocumentRepository()
    service = DocumentService(repository)

    updated = service._persist_document_identity(
        document,
        {"document_title": "员工手册", "aliases": ["手册"]},
    )

    assert updated.document_metadata["existing"] == {"keep": True}
    assert updated.document_metadata["document_identity"]["document_title"] == "员工手册"
    assert repository.updated_data["document_metadata"]["document_identity"]["aliases"] == ["手册"]


def test_identity_persist_failure_marks_parse_failure_path():
    document = _document(7, filename="random.pdf", identity=None)
    repository = FakeDocumentRepository(should_fail=True)
    service = DocumentService(repository)

    try:
        service._persist_document_identity(document, {"document_title": "失败文档"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("persist failure should bubble to parse_document")


def test_parse_document_fails_when_identity_save_fails(monkeypatch):
    document = _document(9, filename="random.pdf", identity=None)
    repository = FakeDocumentRepository(document=document, fail_on_document_metadata=True)
    service = DocumentService(repository)
    monkeypatch.setattr(
        "backend.app.services.document.DocumentPipeline",
        lambda: FakePipeline(),
    )

    with raises(BusinessException):
        service.parse_document(9)

    assert repository.last_parse_status == "failed"
    assert repository.last_parse_message == "文档身份信息保存失败"


def test_retriever_reads_identity_from_orm_document_metadata_after_object_recreated():
    recreated_document = _document(
        8,
        filename="a8f3c9.pdf",
        identity={
            "document_title": "中华人民共和国劳动法",
            "aliases": ["劳动法"],
            "keywords": [],
            "entities": {},
            "summary": "法律文本",
        },
    )

    result = _builder([recreated_document]).build("劳动法", knowledge_base_id=4)

    assert result.candidate_document_ids == [8]


def _identity(
    *,
    filename: str,
    text: str,
    elements: list[DocumentElement],
):
    return DocumentIdentityAnalyzer().analyze(
        parse_result=ParseResult(text=text, elements=elements, metadata={}),
        cleaned_text=text,
        filename=filename,
    )


def _heading(content: str) -> DocumentElement:
    return DocumentElement(type="heading", content=content, section_path=[content])


def _document(document_id: int, *, filename: str, identity: dict | None):
    class Document:
        knowledge_base_id = 4

    document = Document()
    document.id = document_id
    document.original_filename = filename
    document.filename = filename
    document.storage_path = filename
    document.document_metadata = (
        {"document_identity": identity}
        if identity is not None
        else {}
    )
    return document


def _builder(documents):
    class Builder(AutoMetadataFilterBuilder):
        def _load_documents(self, knowledge_base_id):
            return documents

    return Builder()


class FakeDocumentRepository:
    def __init__(
        self,
        should_fail: bool = False,
        document=None,
        fail_on_document_metadata: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.document = document
        self.fail_on_document_metadata = fail_on_document_metadata
        self.updated_data = {}
        self.last_parse_status = None
        self.last_parse_message = None

    def get(self, document_id):
        return self.document

    def update(self, document, data):
        if self.should_fail or (
            self.fail_on_document_metadata and "document_metadata" in data
        ):
            raise RuntimeError("db failed")
        self.updated_data = data
        if "parse_status" in data:
            self.last_parse_status = data["parse_status"]
            self.last_parse_message = data.get("parse_message")
        for key, value in data.items():
            setattr(document, key, value)
        return document


class FakePipeline:
    def run(self, document):
        return SimpleNamespace(
            parse_result=ParseResult(text="员工手册", metadata={}),
            clean_result=CleanResult(
                text="员工手册",
                original_length=4,
                cleaned_length=4,
                metadata={},
            ),
            chunk_result=SimpleNamespace(chunks=[], strategy="fixed", total_chunks=0),
            embedding_result=SimpleNamespace(items=[]),
            vector_store_result=SimpleNamespace(),
            metadata={"document_identity": {"document_title": "员工手册"}},
        )
