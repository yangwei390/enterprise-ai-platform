import pytest
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.neighbor_expansion_step import (
    NeighborExpansionConfig,
    NeighborExpansionStep,
)


class FakeNeighborLookup:
    def __init__(self, chunks: dict[tuple[int, int, int], RerankedChunk]) -> None:
        self.chunks = chunks

    def find_neighbor(
        self,
        *,
        document_id: int,
        knowledge_base_id: int,
        chunk_index: int,
    ) -> RerankedChunk | None:
        return self.chunks.get((knowledge_base_id, document_id, chunk_index))


class FailingNeighborLookup:
    def find_neighbor(
        self,
        *,
        document_id: int,
        knowledge_base_id: int,
        chunk_index: int,
    ) -> RerankedChunk | None:
        raise RuntimeError("neighbor lookup failed")


def _config(
    *,
    enabled: bool = True,
    before: int = 1,
    after: int = 1,
    max_added_chunks: int = 10,
    fail_open: bool = True,
) -> NeighborExpansionConfig:
    return NeighborExpansionConfig(
        enabled=enabled,
        before=before,
        after=after,
        max_added_chunks=max_added_chunks,
        fail_open=fail_open,
    )


def _chunk(
    *,
    chunk_index: int | None,
    document_id: int | None = 9,
    knowledge_base_id: int | None = 4,
    rerank_score: float = 0.9,
) -> RerankedChunk:
    return RerankedChunk(
        id=f"{document_id}_{chunk_index}",
        original_score=rerank_score,
        rerank_score=rerank_score,
        text=f"chunk {chunk_index}",
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_index=chunk_index,
        metadata={
            "source": "law.pdf",
            "document_id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "chunk_index": chunk_index,
            "parser": "PdfParser",
            "cleaner": "BasicTextCleaner",
            "strategy": "fixed",
            "rerank_rank": chunk_index,
            "rerank_provider": "dummy",
            "rerank_model": "dummy-reranker",
        },
    )


def test_neighbor_expansion_disabled_returns_original_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    chunks = [_chunk(chunk_index=2)]
    context = RetrieverPipelineContext(query="测试", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.neighbor_expansion_step."
        "get_neighbor_expansion_config",
        lambda: _config(enabled=False),
    )

    result = NeighborExpansionStep(lookup=FakeNeighborLookup({})).run(context)

    assert result.reranked_chunks == chunks
    assert result.metadata["neighbor_expansion"]["enabled"] is False
    assert result.metadata["neighbor_expansion"]["added_chunk_count"] == 0


def test_neighbor_expansion_adds_before_and_after_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk_1 = _chunk(chunk_index=1)
    chunk_2 = _chunk(chunk_index=2)
    chunk_3 = _chunk(chunk_index=3)
    context = RetrieverPipelineContext(query="测试", reranked_chunks=[chunk_2])
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.neighbor_expansion_step."
        "get_neighbor_expansion_config",
        lambda: _config(before=1, after=1),
    )

    result = NeighborExpansionStep(
        lookup=FakeNeighborLookup(
            {
                (4, 9, 1): chunk_1,
                (4, 9, 3): chunk_3,
            }
        )
    ).run(context)

    assert [chunk.chunk_index for chunk in result.reranked_chunks] == [1, 2, 3]
    assert result.metadata["neighbor_expansion"]["added_chunk_count"] == 2
    assert result.reranked_chunks[0].metadata["neighbor_expanded"] is True
    assert result.reranked_chunks[0].metadata["neighbor_position"] == "before"
    assert result.reranked_chunks[2].metadata["neighbor_position"] == "after"


def test_neighbor_expansion_deduplicates_chunks(monkeypatch: pytest.MonkeyPatch):
    chunk_1 = _chunk(chunk_index=1)
    chunk_2 = _chunk(chunk_index=2)
    context = RetrieverPipelineContext(query="测试", reranked_chunks=[chunk_1, chunk_2])
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.neighbor_expansion_step."
        "get_neighbor_expansion_config",
        lambda: _config(before=1, after=0),
    )

    result = NeighborExpansionStep(
        lookup=FakeNeighborLookup({(4, 9, 1): chunk_1})
    ).run(context)

    assert [chunk.chunk_index for chunk in result.reranked_chunks] == [1, 2]
    assert result.metadata["neighbor_expansion"]["added_chunk_count"] == 0


def test_neighbor_expansion_skips_chunk_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk = _chunk(chunk_index=None, document_id=None)
    context = RetrieverPipelineContext(query="测试", reranked_chunks=[chunk])
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.neighbor_expansion_step."
        "get_neighbor_expansion_config",
        lambda: _config(),
    )

    result = NeighborExpansionStep(lookup=FakeNeighborLookup({})).run(context)

    assert result.reranked_chunks == [chunk]
    assert result.metadata["neighbor_expansion"]["skipped_chunk_count"] == 1


def test_neighbor_expansion_fail_open(monkeypatch: pytest.MonkeyPatch):
    chunks = [_chunk(chunk_index=2)]
    context = RetrieverPipelineContext(query="测试", reranked_chunks=chunks)
    monkeypatch.setattr(
        "backend.app.retrievers.pipeline.steps.neighbor_expansion_step."
        "get_neighbor_expansion_config",
        lambda: _config(fail_open=True),
    )

    result = NeighborExpansionStep(lookup=FailingNeighborLookup()).run(context)

    metadata = result.metadata["neighbor_expansion"]
    assert result.reranked_chunks == chunks
    assert metadata["failed"] is True
    assert "neighbor lookup failed" in metadata["error"]
    assert metadata["output_chunk_count"] == len(chunks)
