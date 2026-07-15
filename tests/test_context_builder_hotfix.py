from backend.app.context import BasicContextBuilder, ContextBuildRequest
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.document_router import RoutingResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.pipeline import RetrieverPipeline
from backend.app.retrievers.pipeline.steps.context_build_step import ContextBuildStep
from backend.app.retrievers.strategy import RetrievalStrategy


def test_context_builder_deduplicates_same_chunk_id():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=1, score=0.7, text="same"),
                _chunk("a", document_id=1, chunk_index=1, score=0.9, text="same"),
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
        )
    )

    assert result.total_chunks == 1
    assert result.chunks[0].score == 0.9
    assert result.deduplicated_count == 1


def test_context_builder_deduplicates_neighbor_duplicate_by_position():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=2, score=0.9, text="正文"),
                _chunk(
                    "neighbor-a",
                    document_id=1,
                    chunk_index=2,
                    score=0.8,
                    text="正文",
                    metadata={"neighbor_expanded": True},
                ),
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
        )
    )

    assert result.total_chunks == 1
    assert result.deduplicated_count == 1


def test_context_builder_restores_structure_order_within_document():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("c", document_id=1, chunk_index=3, score=0.99, text="third"),
                _chunk("a", document_id=1, chunk_index=1, score=0.7, text="first"),
                _chunk("b", document_id=1, chunk_index=2, score=0.8, text="second"),
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
        )
    )

    assert [chunk.id for chunk in result.chunks] == ["a", "b", "c"]


def test_context_builder_multi_document_quota_preserves_target_evidence():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=1, score=0.99),
                _chunk("b", document_id=1, chunk_index=2, score=0.98),
                _chunk("c", document_id=1, chunk_index=3, score=0.97),
                _chunk("d", document_id=2, chunk_index=1, score=0.6),
            ],
            route_type="MULTI_DOCUMENT",
            strategy="MULTI_DOCUMENT",
            target_document_ids=[1, 2],
            max_chunks=3,
            multi_document_max_per_document=2,
        )
    )

    document_ids = [chunk.document_id for chunk in result.chunks]
    assert 2 in document_ids
    assert document_ids.count(1) <= 2
    assert result.document_groups[0]["document_id"] == 1
    assert result.document_groups[1]["document_id"] == 2


def test_context_builder_backfills_when_evidence_is_insufficient():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=1, score=0.99),
                _chunk("b", document_id=1, chunk_index=2, score=0.98),
            ],
            route_type="MULTI_DOCUMENT",
            strategy="MULTI_DOCUMENT",
            target_document_ids=[1, 2],
            max_chunks=2,
            multi_document_max_per_document=1,
        )
    )

    assert [chunk.id for chunk in result.chunks] == ["a", "b"]
    assert result.total_chunks == 2


def test_context_builder_respects_character_budget_without_hard_truncating():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=1, score=0.9, text="short text"),
                _chunk("b", document_id=1, chunk_index=2, score=0.8, text="x" * 500),
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
            max_context_chars=250,
        )
    )

    assert [chunk.id for chunk in result.chunks] == ["a"]
    assert result.truncated is True
    assert result.skipped_count == 1
    assert "x" * 100 not in result.context_text


def test_context_builder_respects_chunk_count_budget():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk("a", document_id=1, chunk_index=1, score=0.9),
                _chunk("b", document_id=1, chunk_index=2, score=0.8),
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
            max_chunks=1,
        )
    )

    assert result.total_chunks == 1
    assert result.skipped_count == 1


def test_context_builder_preserves_citations():
    result = _builder().build(
        ContextBuildRequest(
            query="q",
            chunks=[
                _chunk(
                    "a",
                    document_id=1,
                    chunk_index=1,
                    score=0.9,
                    metadata={"source": "doc.pdf", "citations": ["c1"]},
                )
            ],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
        )
    )

    assert result.citations[0]["source"] == "doc.pdf"
    assert result.citations[0]["metadata"]["citations"] == ["c1"]


def test_context_builder_fail_open_uses_internal_simple_selection():
    class BrokenBuilder(BasicContextBuilder):
        def _build(self, *args, **kwargs):
            raise RuntimeError("context builder broken")

    result = BrokenBuilder().build(
        ContextBuildRequest(
            query="q",
            chunks=[_chunk("a", document_id=1, chunk_index=1, score=0.9)],
            route_type="DOCUMENT",
            strategy="DOCUMENT",
            target_document_ids=[1],
        )
    )

    assert "Document:" in result.context_text
    assert result.metadata["failed"] is True
    assert result.metadata["error"] == "context builder broken"


def test_context_build_step_writes_unified_context_builder_trace():
    context = RetrieverPipelineContext(query="q")
    context.reranked_chunks = [_chunk("a", document_id=1, chunk_index=1, score=0.9)]
    context.routing_result = RoutingResult(route_type="DOCUMENT", target_document_ids=[1])
    context.retrieval_strategy = RetrievalStrategy(
        strategy="DOCUMENT",
        route_type="DOCUMENT",
        document_ids=[1],
        global_budget=5,
    )

    result = ContextBuildStep().run(context)

    assert "Document:" in result.context_text
    assert "context_builder" in result.metadata
    assert "context_builder" + "_v2" not in result.metadata
    assert result.metadata["context_builder"]["route_type"] == "DOCUMENT"


def test_pipeline_order_keeps_context_builder_after_neighbor_before_compression():
    pipeline = RetrieverPipeline()

    step_names = [step.__class__.__name__ for step in pipeline.steps]

    assert step_names.index("NeighborExpansionStep") < step_names.index(
        "ContextBuildStep"
    )
    assert step_names.index("ContextBuildStep") < step_names.index(
        "ContextCompressionStep"
    )


def _builder() -> BasicContextBuilder:
    return BasicContextBuilder()


def _chunk(
    chunk_id: str,
    *,
    document_id: int,
    chunk_index: int,
    score: float,
    text: str | None = None,
    metadata: dict | None = None,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk_id,
        original_score=score,
        rerank_score=score,
        text=text or f"text {chunk_id}",
        document_id=document_id,
        knowledge_base_id=1,
        chunk_index=chunk_index,
        metadata={
            "source": f"doc-{document_id}.pdf",
            "rerank_score": score,
            "rerank_rank": 1,
            **(metadata or {}),
        },
    )
