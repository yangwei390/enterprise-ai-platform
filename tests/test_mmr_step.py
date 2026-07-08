import pytest
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.mmr_step import (
    MMRConfig,
    MMRStep,
    text_similarity,
)


def _config(
    *,
    enabled: bool = True,
    lambda_value: float = 0.7,
    top_k: int = 5,
    min_score: float = 0.0,
    fail_open: bool = True,
    similarity_threshold: float = 0.85,
) -> MMRConfig:
    return MMRConfig(
        enabled=enabled,
        lambda_value=lambda_value,
        top_k=top_k,
        min_score=min_score,
        fail_open=fail_open,
        similarity_threshold=similarity_threshold,
    )


def _chunk(
    *,
    chunk_id: str,
    text: str,
    rerank_score: float,
    chunk_index: int,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk_id,
        original_score=rerank_score,
        rerank_score=rerank_score,
        text=text,
        document_id=10,
        knowledge_base_id=20,
        chunk_index=chunk_index,
        metadata={
            "source": "source.txt",
            "document_id": 10,
            "knowledge_base_id": 20,
            "chunk_index": chunk_index,
            "rerank_score": rerank_score,
            "rerank_rank": chunk_index,
            "rerank_provider": "dummy",
            "rerank_model": "dummy-reranker",
            "custom": f"custom-{chunk_id}",
        },
    )


def test_mmr_disabled_returns_original_chunks(monkeypatch: pytest.MonkeyPatch):
    chunks = [_chunk(chunk_id="1", text="文本一", rerank_score=0.9, chunk_index=1)]
    context = RetrieverPipelineContext(query="测试", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.get_mmr_config",
        lambda: _config(enabled=False),
    )

    result = MMRStep().run(context)

    assert result.reranked_chunks == chunks
    assert result.metadata["mmr"]["enabled"] is False
    assert result.metadata["mmr"]["selected_chunk_count"] == 1


def test_mmr_selects_top_k_chunks(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        _chunk(
            chunk_id=str(index),
            text=f"完全不同的文本 {index}",
            rerank_score=1 - index * 0.01,
            chunk_index=index,
        )
        for index in range(6)
    ]
    context = RetrieverPipelineContext(query="测试", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.get_mmr_config",
        lambda: _config(top_k=3),
    )

    result = MMRStep().run(context)

    assert len(result.reranked_chunks) == 3
    assert result.metadata["mmr"]["selected_chunk_count"] == 3
    assert result.metadata["mmr"]["removed_chunk_count"] == 3


def test_mmr_preserves_metadata(monkeypatch: pytest.MonkeyPatch):
    chunk = _chunk(
        chunk_id="1",
        text="劳动法第二章促进就业",
        rerank_score=0.9,
        chunk_index=1,
    )
    context = RetrieverPipelineContext(query="劳动法", reranked_chunks=[chunk])
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.get_mmr_config",
        lambda: _config(top_k=1),
    )

    result = MMRStep().run(context)
    metadata = result.reranked_chunks[0].metadata

    assert metadata["document_id"] == 10
    assert metadata["knowledge_base_id"] == 20
    assert metadata["chunk_index"] == 1
    assert metadata["source"] == "source.txt"
    assert metadata["rerank_score"] == 0.9
    assert metadata["custom"] == "custom-1"
    assert metadata["mmr_selected"] is True
    assert metadata["mmr_rank"] == 1


def test_mmr_reduces_duplicate_chunks(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        _chunk(
            chunk_id="1",
            text="劳动法第二章促进就业内容",
            rerank_score=0.9,
            chunk_index=1,
        ),
        _chunk(
            chunk_id="2",
            text="劳动法第二章促进就业内容",
            rerank_score=0.89,
            chunk_index=2,
        ),
        _chunk(
            chunk_id="3",
            text="劳动合同解除条件",
            rerank_score=0.7,
            chunk_index=3,
        ),
    ]
    context = RetrieverPipelineContext(query="劳动法", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.get_mmr_config",
        lambda: _config(top_k=3, similarity_threshold=0.85),
    )

    result = MMRStep().run(context)

    assert [chunk.id for chunk in result.reranked_chunks] == ["1", "3"]
    assert result.metadata["mmr"]["removed_chunk_count"] == 1


def test_mmr_fail_open_returns_original_chunks(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        _chunk(
            chunk_id="1",
            text="劳动法第二章促进就业",
            rerank_score=0.9,
            chunk_index=1,
        ),
        _chunk(
            chunk_id="2",
            text="劳动合同解除条件",
            rerank_score=0.8,
            chunk_index=2,
        ),
    ]
    context = RetrieverPipelineContext(query="劳动法", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.get_mmr_config",
        lambda: _config(fail_open=True),
    )
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.mmr_step.text_similarity",
        lambda left, right: (_ for _ in ()).throw(RuntimeError("similarity failed")),
    )

    result = MMRStep().run(context)

    assert result.reranked_chunks == chunks
    assert result.metadata["mmr"]["failed"] is True
    assert "similarity failed" in result.metadata["mmr"]["error"]


def test_text_similarity_identical_is_one():
    assert text_similarity("劳动法第二章", "劳动法第二章") == 1


def test_text_similarity_empty_is_zero():
    assert text_similarity("", "劳动法第二章") == 0
    assert text_similarity("劳动法第二章", "") == 0
