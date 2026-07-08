import pytest
from backend.app.context.base import ContextChunk
from backend.app.context_compression import CompressionInput
from backend.app.context_compression.config import ContextCompressionConfig
from backend.app.context_compression.factory import ContextCompressorFactory
from backend.app.context_compression.llm_compressor import LLMContextCompressor
from backend.app.context_compression.rule_based_compressor import (
    RuleBasedContextCompressor,
)
from backend.app.llms import LLMRequest, LLMResponse
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.context_compression_step import (
    ContextCompressionStep,
)


class ExplodingCompressor:
    def compress(self, input: CompressionInput):
        raise RuntimeError("compression exploded")


class FakeLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.requests: list[LLMRequest] = []

    def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        answer = self.answers[len(self.requests) - 1]
        return LLMResponse(answer=answer, model=request.model or "fake-model")


def _context_chunk(
    *,
    chunk_id: str,
    text: str,
    rerank_score: float,
    rerank_rank: int,
) -> ContextChunk:
    return ContextChunk(
        id=chunk_id,
        text=text,
        document_id=10,
        knowledge_base_id=20,
        chunk_index=rerank_rank,
        score=rerank_score,
        source="source.txt",
        metadata={
            "source": "source.txt",
            "document_id": 10,
            "knowledge_base_id": 20,
            "chunk_index": rerank_rank,
            "score": rerank_score,
            "rerank_score": rerank_score,
            "rerank_rank": rerank_rank,
            "rerank_provider": "dummy",
            "rerank_model": "dummy-reranker",
            "custom": f"custom-{chunk_id}",
        },
    )


def test_compression_disabled_returns_original_context(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        _context_chunk(
            chunk_id="chunk-1",
            text="原始上下文内容",
            rerank_score=0.9,
            rerank_rank=1,
        )
    ]
    original_context = "[来源: source.txt, 文档ID: 10, Chunk: 1]\n原始上下文内容"
    context = RetrieverPipelineContext(
        query="测试问题",
        context_chunks=chunks,
        context_text=original_context,
    )

    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.context_compression_step."
        "get_context_compression_config",
        lambda: ContextCompressionConfig(
            enabled=False,
            provider="rule_based",
            max_chars=6000,
            max_chunk_chars=1200,
            fail_open=True,
        ),
    )

    result = ContextCompressionStep().run(context)

    assert result.context_text == original_context
    assert result.context_chunks == chunks
    assert result.context_chunks[0].metadata["rerank_provider"] == "dummy"
    assert result.metadata["context_compression"]["enabled"] is False
    assert result.metadata["context_compression"]["compressed_chunk_count"] == 1


def test_rule_based_compression_respects_max_chars():
    low_score_chunk = _context_chunk(
        chunk_id="low",
        text="普通内容。" * 40,
        rerank_score=0.1,
        rerank_rank=2,
    )
    high_score_chunk = _context_chunk(
        chunk_id="high",
        text="劳动法第二章讲的是促进就业。" * 40,
        rerank_score=0.99,
        rerank_rank=1,
    )
    compressor = RuleBasedContextCompressor(max_chunk_chars=50)

    result = compressor.compress(
        CompressionInput(
            query="劳动法第二章讲的什么？",
            chunks=[low_score_chunk, high_score_chunk],
            max_chars=120,
        )
    )

    assert result.compressed_chars <= 120
    assert result.compressed_chunks[0].id == "high"
    assert len(result.compressed_chunks[0].text) <= 50
    assert result.compressed_chunks[0].metadata["source"] == "source.txt"
    assert result.compressed_chunks[0].metadata["document_id"] == 10
    assert result.compressed_chunks[0].metadata["knowledge_base_id"] == 20
    assert result.compressed_chunks[0].metadata["chunk_index"] == 1
    assert result.compressed_chunks[0].metadata["score"] == 0.99
    assert result.compressed_chunks[0].metadata["rerank_score"] == 0.99
    assert result.compressed_chunks[0].metadata["rerank_rank"] == 1
    assert result.compressed_chunks[0].metadata["rerank_provider"] == "dummy"
    assert result.compressed_chunks[0].metadata["rerank_model"] == "dummy-reranker"
    assert result.compressed_chunks[0].metadata["custom"] == "custom-high"


def test_compression_fail_open_returns_original_context(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        _context_chunk(
            chunk_id="chunk-1",
            text="原始上下文内容",
            rerank_score=0.9,
            rerank_rank=1,
        )
    ]
    original_context = "[来源: source.txt, 文档ID: 10, Chunk: 1]\n原始上下文内容"
    context = RetrieverPipelineContext(
        query="测试问题",
        context_chunks=chunks,
        context_text=original_context,
    )

    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.context_compression_step."
        "get_context_compression_config",
        lambda: ContextCompressionConfig(
            enabled=True,
            provider="rule_based",
            max_chars=6000,
            max_chunk_chars=1200,
            fail_open=True,
        ),
    )
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.context_compression_step."
        "ContextCompressorFactory.get_compressor",
        lambda provider: ExplodingCompressor(),
    )

    result = ContextCompressionStep().run(context)
    metadata = result.metadata["context_compression"]

    assert result.context_text == original_context
    assert result.context_chunks == chunks
    assert result.context_chunks[0].metadata["rerank_model"] == "dummy-reranker"
    assert metadata["failed"] is True
    assert "compression exploded" in metadata["error"]
    assert metadata["original_chars"] == len(original_context)
    assert metadata["compressed_chars"] == len(original_context)
    assert metadata["original_chunk_count"] == 1
    assert metadata["compressed_chunk_count"] == 1


def test_llm_compressor_preserves_metadata():
    fake_llm = FakeLLM(["劳动法第二章讲的是促进就业。"])
    chunk = _context_chunk(
        chunk_id="chunk-1",
        text="劳动法第二章讲的是促进就业。其他无关内容。" * 5,
        rerank_score=0.9,
        rerank_rank=1,
    )
    compressor = LLMContextCompressor(
        model="qwen-turbo",
        temperature=0,
        max_chunk_chars=1200,
        max_calls=8,
        llm=fake_llm,
    )

    result = compressor.compress(
        CompressionInput(
            query="劳动法第二章讲什么？",
            chunks=[chunk],
            max_chars=6000,
        )
    )

    metadata = result.compressed_chunks[0].metadata
    assert metadata["document_id"] == 10
    assert metadata["knowledge_base_id"] == 20
    assert metadata["chunk_index"] == 1
    assert metadata["source"] == "source.txt"
    assert metadata["rerank_score"] == 0.9
    assert metadata["rerank_rank"] == 1
    assert metadata["rerank_provider"] == "dummy"
    assert metadata["rerank_model"] == "dummy-reranker"
    assert metadata["context_compressed"] is True
    assert metadata["context_compression_provider"] == "llm"
    assert metadata["context_compression_original_chars"] == len(chunk.text)
    assert metadata["context_compression_compressed_chars"] == len(
        result.compressed_chunks[0].text
    )


def test_llm_compressor_skips_irrelevant_chunk():
    fake_llm = FakeLLM(["相关内容", ""])
    relevant_chunk = _context_chunk(
        chunk_id="relevant",
        text="劳动法第二章讲的是促进就业。",
        rerank_score=0.9,
        rerank_rank=1,
    )
    irrelevant_chunk = _context_chunk(
        chunk_id="irrelevant",
        text="Android ViewModel 内容。",
        rerank_score=0.8,
        rerank_rank=2,
    )
    compressor = LLMContextCompressor(
        model="qwen-turbo",
        max_calls=8,
        llm=fake_llm,
    )

    result = compressor.compress(
        CompressionInput(
            query="劳动法第二章讲什么？",
            chunks=[relevant_chunk, irrelevant_chunk],
            max_chars=6000,
        )
    )

    assert [chunk.id for chunk in result.compressed_chunks] == ["relevant"]
    assert result.metadata["llm_empty_outputs"] == 1
    assert result.skipped_chunk_count == 1


def test_llm_compressor_fail_open_when_all_empty():
    fake_llm = FakeLLM(["", ""])
    chunks = [
        _context_chunk(
            chunk_id="chunk-1",
            text="无关内容 1",
            rerank_score=0.9,
            rerank_rank=1,
        ),
        _context_chunk(
            chunk_id="chunk-2",
            text="无关内容 2",
            rerank_score=0.8,
            rerank_rank=2,
        ),
    ]
    compressor = LLMContextCompressor(
        model="qwen-turbo",
        max_calls=8,
        llm=fake_llm,
    )

    result = compressor.compress(
        CompressionInput(
            query="劳动法第二章讲什么？",
            chunks=chunks,
            max_chars=6000,
        )
    )

    assert result.compressed_chunks == chunks
    assert result.compressed_chunk_count == len(chunks)
    assert result.metadata["fallback_used"] is True
    assert result.metadata["llm_empty_outputs"] == 2
    assert result.metadata["context_text"]


def test_factory_creates_llm_compressor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "backend.app.context_compression.factory.get_context_compression_config",
        lambda: ContextCompressionConfig(
            enabled=True,
            provider="llm",
            max_chars=6000,
            max_chunk_chars=1200,
            fail_open=True,
            llm_model="qwen-turbo",
            llm_temperature=0,
            llm_max_chars_per_chunk=1200,
            llm_timeout_seconds=30,
            llm_max_calls=8,
        ),
    )

    compressor = ContextCompressorFactory.get_compressor()

    assert isinstance(compressor, LLMContextCompressor)


def test_llm_compressor_respects_max_calls():
    fake_llm = FakeLLM(["high compressed", "middle compressed"])
    low_chunk = _context_chunk(
        chunk_id="low",
        text="low",
        rerank_score=0.1,
        rerank_rank=3,
    )
    high_chunk = _context_chunk(
        chunk_id="high",
        text="high",
        rerank_score=0.9,
        rerank_rank=1,
    )
    middle_chunk = _context_chunk(
        chunk_id="middle",
        text="middle",
        rerank_score=0.5,
        rerank_rank=2,
    )
    compressor = LLMContextCompressor(
        model="qwen-turbo",
        max_calls=2,
        llm=fake_llm,
    )

    result = compressor.compress(
        CompressionInput(
            query="测试",
            chunks=[low_chunk, high_chunk, middle_chunk],
            max_chars=6000,
        )
    )

    assert result.metadata["llm_calls"] == 2
    assert len(fake_llm.requests) == 2
    assert result.metadata["llm_skipped_chunks"] == 1
    assert [chunk.id for chunk in result.compressed_chunks] == ["high", "middle"]
