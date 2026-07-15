import inspect

from backend.app.config.settings import settings
from backend.app.rerankers import RerankedChunk
from backend.app.rerankers.v2 import RerankerV2Enhancer
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.document_router import RoutingResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.pipeline import RetrieverPipeline
from backend.app.retrievers.pipeline.steps.rerank_step import RerankStep
from backend.app.retrievers.planning import RetrievalConstraint, RetrievalPlan
from backend.app.retrievers.strategy import RetrievalStrategy


def test_document_strategy_promotes_exact_structure_match_without_score_override():
    context = _context(strategy="DOCUMENT", route_type="DOCUMENT", document_ids=[1])
    context.retrieval_plan = _plan(
        [
            RetrievalConstraint(
                field="chapter_number",
                operator="eq",
                value=2,
                applied=True,
            )
        ]
    )
    chunks = [
        _reranked("a", document_id=1, score=0.9, metadata={"chapter_number": 1}),
        _reranked("b", document_id=1, score=0.89, metadata={"chapter_number": 2}),
    ]

    result = RerankerV2Enhancer().enhance(
        chunks=chunks,
        context=context,
        diversity_enabled=True,
        max_per_document=3,
        min_documents=2,
    )

    assert [chunk.id for chunk in result.chunks] == ["b", "a"]
    assert result.chunks[0].rerank_score == 0.89
    assert result.chunks[0].metadata["structure_match"] is True
    assert result.chunks[0].metadata["identity_match"] is True
    assert result.chunks[0].metadata["original_fusion_score"] == 0.89
    assert result.metadata["structure_boost_count"] == 1


def test_global_strategy_preserves_model_order():
    context = _context(strategy="GLOBAL", route_type="KNOWLEDGE_BASE")
    chunks = [
        _reranked("a", document_id=2, score=0.8),
        _reranked("b", document_id=1, score=0.7),
    ]

    result = RerankerV2Enhancer().enhance(
        chunks=chunks,
        context=context,
        diversity_enabled=True,
        max_per_document=1,
        min_documents=2,
    )

    assert [chunk.id for chunk in result.chunks] == ["a", "b"]
    assert result.metadata["diversity_applied"] is False


def test_multi_document_applies_max_per_document_quota_and_backfills():
    context = _context(
        strategy="MULTI_DOCUMENT",
        route_type="MULTI_DOCUMENT",
        document_ids=[1, 2],
        top_k=4,
    )
    chunks = [
        _reranked("a", document_id=1, score=0.99),
        _reranked("b", document_id=1, score=0.98),
        _reranked("c", document_id=1, score=0.97),
        _reranked("d", document_id=2, score=0.7),
        _reranked("e", document_id=1, score=0.69),
    ]

    result = RerankerV2Enhancer().enhance(
        chunks=chunks,
        context=context,
        diversity_enabled=True,
        max_per_document=2,
        min_documents=2,
    )

    assert [chunk.document_id for chunk in result.chunks] == [1, 1, 2, 1]
    assert result.metadata["diversity_applied"] is True
    assert result.metadata["rejected_by_quota_count"] == 2
    assert result.metadata["backfilled_count"] == 1


def test_multi_document_does_not_force_average_when_candidates_are_insufficient():
    context = _context(
        strategy="MULTI_DOCUMENT",
        route_type="MULTI_DOCUMENT",
        document_ids=[1],
        top_k=3,
    )
    chunks = [
        _reranked("a", document_id=1, score=0.99),
        _reranked("b", document_id=1, score=0.98),
        _reranked("c", document_id=1, score=0.97),
    ]

    result = RerankerV2Enhancer().enhance(
        chunks=chunks,
        context=context,
        diversity_enabled=True,
        max_per_document=1,
        min_documents=2,
    )

    assert [chunk.id for chunk in result.chunks] == ["a", "b", "c"]
    assert result.metadata["rejected_by_quota_count"] == 0


def test_rerank_step_fail_open_preserves_original_reranker_order(monkeypatch):
    class BrokenEnhancer:
        def enhance(self, **kwargs):
            raise RuntimeError("enhancer broken")

    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dummy")
    monkeypatch.setattr(settings, "RERANK_MULTI_DOCUMENT_FAIL_OPEN", True)
    context = _context(
        strategy="MULTI_DOCUMENT",
        route_type="MULTI_DOCUMENT",
        document_ids=[1, 2],
        top_k=2,
    )
    context.fused_chunks = [
        _retrieved("a", document_id=1, score=0.9),
        _retrieved("b", document_id=2, score=0.8),
    ]

    result = RerankStep(enhancer=BrokenEnhancer()).run(context)

    assert [chunk.id for chunk in result.reranked_chunks] == ["a", "b"]
    assert result.metadata["reranker_v2"]["failed"] is True
    assert result.metadata["reranker_v2"]["error"] == "enhancer broken"


def test_reranker_v2_has_no_domain_hardcoded_rules():
    source = inspect.getsource(RerankerV2Enhancer)

    for forbidden in ("劳动法", "劳动合同", "员工手册", "促进就业"):
        assert forbidden not in source


def test_pipeline_order_keeps_rerank_before_mmr_and_after_fusion():
    pipeline = RetrieverPipeline()

    step_names = [step.__class__.__name__ for step in pipeline.steps]

    assert step_names.index("FusionStep") < step_names.index("RerankStep")
    assert step_names.index("RerankStep") < step_names.index("MMRStep")


def _context(
    *,
    strategy: str,
    route_type: str,
    document_ids: list[int] | None = None,
    top_k: int = 5,
) -> RetrieverPipelineContext:
    context = RetrieverPipelineContext(query="query", top_k=top_k)
    context.routing_result = RoutingResult(
        route_type=route_type,
        target_document_ids=document_ids or [],
    )
    context.retrieval_strategy = RetrievalStrategy(
        strategy=strategy,
        route_type=route_type,
        document_ids=document_ids or [],
        global_budget=top_k,
        per_document_budget=2 if strategy == "MULTI_DOCUMENT" else top_k,
    )
    return context


def _plan(constraints: list[RetrievalConstraint]) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="query",
        rewritten_query="query",
        intent="structured",
        strategy="structured_hybrid",
        constraints=constraints,
        use_structure_filter=True,
    )


def _reranked(
    chunk_id: str,
    *,
    document_id: int,
    score: float,
    metadata: dict | None = None,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk_id,
        original_score=score,
        rerank_score=score,
        text=chunk_id,
        document_id=document_id,
        knowledge_base_id=1,
        chunk_index=1,
        metadata={
            **(metadata or {}),
            "rerank_score": score,
            "rerank_rank": 1,
        },
    )


def _retrieved(chunk_id: str, *, document_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        score=score,
        text=chunk_id,
        document_id=document_id,
        knowledge_base_id=1,
        chunk_index=1,
        metadata={"fusion_score": score},
    )
