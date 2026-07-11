from dataclasses import dataclass

import pytest
from backend.app.api.document import get_document_service
from backend.app.api.document import router as document_router
from backend.app.chunkers import Chunk, ChunkResult, FixedChunker, LegalStructureChunker
from backend.app.chunkers.markdown import MarkdownChunker
from backend.app.documents.structures.legal import LegalStructureParser
from backend.app.documents.structures.markdown import MarkdownStructureParser
from backend.app.embeddings.base import BaseEmbedding, EmbeddingBatchError
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.exceptions import BusinessException
from backend.app.exceptions.handlers import register_exception_handlers
from backend.app.pipeline import DocumentPipeline
from backend.app.pipeline.base import PipelineContext, PipelineStep
from fastapi import FastAPI
from fastapi.testclient import TestClient


class BatchSpyEmbedding(BaseEmbedding):
    model_name = "batch-spy"
    provider_name = "test"
    dimension = 1

    def __init__(self, *, batch_size: int = 10, fail_batch: int | None = None) -> None:
        self.config = EmbeddingConfig(batch_size=batch_size)
        self.fail_batch = fail_batch
        self.batches: list[list[str]] = []

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.fail_batch == len(self.batches):
            raise RuntimeError("batch failed")
        return [[float(text.removeprefix("text-"))] for text in texts]


def _chunks(count: int) -> list[Chunk]:
    return [
        Chunk(
            document_id=1,
            knowledge_base_id=1,
            chunk_index=index,
            text=f"text-{index}",
            start_offset=index,
            end_offset=index + 1,
            metadata={"chunk_index": index},
        )
        for index in range(count)
    ]


def test_embedding_batches_25_texts_as_10_10_5() -> None:
    embedding = BatchSpyEmbedding(batch_size=10)

    result = embedding.embed_chunks(_chunks(25))

    assert [len(batch) for batch in embedding.batches] == [10, 10, 5]
    assert result.metadata["total_texts"] == 25
    assert result.metadata["batch_size"] == 10
    assert result.metadata["batch_count"] == 3


def test_embedding_vectors_preserve_input_order() -> None:
    embedding = BatchSpyEmbedding(batch_size=10)

    result = embedding.embed_chunks(_chunks(25))

    assert [item.vector[0] for item in result.items] == [float(index) for index in range(25)]
    assert [item.chunk_index for item in result.items] == list(range(25))


def test_embedding_batch_failure_fails_whole_request() -> None:
    embedding = BatchSpyEmbedding(batch_size=10, fail_batch=2)

    with pytest.raises(EmbeddingBatchError) as exc_info:
        embedding.embed_chunks(_chunks(25))

    assert exc_info.value.metadata["failed_batch"] == 2
    assert exc_info.value.metadata["batch_count"] == 3


def test_embedding_batch_failure_does_not_return_partial_vectors() -> None:
    embedding = BatchSpyEmbedding(batch_size=10, fail_batch=2)

    with pytest.raises(EmbeddingBatchError):
        result = embedding.embed_chunks(_chunks(25))
        pytest.fail(f"unexpected partial embedding result: {result!r}")


class SeedChunksStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        context.chunk_result = ChunkResult(
            strategy="fixed",
            chunk_size=100,
            chunk_overlap=0,
            chunks=_chunks(3),
            total_chunks=3,
        )
        return context


class FailingEmbeddingStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        raise EmbeddingBatchError(
            "Embedding批次调用失败",
            {
                "total_texts": 3,
                "batch_size": 10,
                "batch_count": 1,
                "failed_batch": 1,
            },
        )


class FlagStep(PipelineStep):
    def __init__(self) -> None:
        self.called = False

    def run(self, context: PipelineContext) -> PipelineContext:
        self.called = True
        return context


@dataclass
class PipelineDocument:
    id: int = 1
    knowledge_base_id: int = 1


def test_document_pipeline_embedding_failure_stops_later_steps() -> None:
    vector_step = FlagStep()
    bm25_step = FlagStep()
    pipeline = DocumentPipeline()
    pipeline.steps = [
        SeedChunksStep(),
        FailingEmbeddingStep(),
        vector_step,
        bm25_step,
    ]

    with pytest.raises(EmbeddingBatchError):
        pipeline.run(PipelineDocument())

    assert vector_step.called is False
    assert bm25_step.called is False


class FailingDocumentService:
    def parse_document(self, document_id: int) -> dict:
        raise BusinessException(41003, "文档解析失败")


class SuccessfulDocumentService:
    def parse_document(self, document_id: int) -> dict:
        return {
            "document_id": document_id,
            "text_length": 10,
            "preview": "hello",
            "page_count": 1,
            "metadata": {},
            "original_length": 10,
            "cleaned_length": 10,
            "cleaner_metadata": {},
            "chunk_strategy": "fixed",
            "chunk_size": 100,
            "chunk_overlap": 0,
            "total_chunks": 1,
            "chunks_preview": [],
            "embedding_model": "dummy",
            "embedding_dimension": 8,
            "total_embeddings": 1,
            "embeddings_preview": [],
            "vector_collection": "documents",
            "vector_total_records": 1,
            "vector_ids_preview": ["1_0"],
        }


def _document_api(service) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(document_router)
    app.dependency_overrides[get_document_service] = lambda: service
    return TestClient(app)


def test_parse_failure_api_does_not_return_200() -> None:
    response = _document_api(FailingDocumentService()).post("/documents/1/parse")

    assert response.status_code == 500
    assert response.json()["code"] == 41003
    assert response.json()["message"] == "文档解析失败"


def test_parse_success_api_response_remains_compatible() -> None:
    response = _document_api(SuccessfulDocumentService()).post("/documents/1/parse")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["document_id"] == 1
    assert body["data"]["chunk_strategy"] == "fixed"


def test_fixed_legal_markdown_chunking_still_work() -> None:
    fixed = FixedChunker(chunk_size=20, chunk_overlap=0).chunk("hello world")
    assert fixed.total_chunks >= 1

    legal_text = (
        "中华人民共和国示例法\n"
        "第一章 总则\n"
        "第一条 第一条内容。\n"
        "第二章 促进事项\n"
        "第十条 第十条内容。"
    )
    legal_structure = LegalStructureParser().parse(legal_text, {"document_id": 1})
    legal = LegalStructureChunker().chunk(legal_text, {"_document_structure": legal_structure})
    assert legal.total_chunks >= 1
    assert any(chunk.metadata.get("document_type") == "legal" for chunk in legal.chunks)

    markdown_text = "# RAG\n\n## Retriever\n\nBM25 content"
    markdown_structure = MarkdownStructureParser().parse(markdown_text, {"document_id": 1})
    markdown = MarkdownChunker().chunk(markdown_text, {"_document_structure": markdown_structure})
    assert markdown.total_chunks >= 1
    assert any(chunk.metadata.get("chunk_strategy") == "markdown" for chunk in markdown.chunks)
