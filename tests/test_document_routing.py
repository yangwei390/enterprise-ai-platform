from backend.app.config.settings import settings
from backend.app.query import QueryRewriteResult
from backend.app.retrievers.document_router import DocumentRouter
from backend.app.retrievers.metadata_filter import AutoMetadataFilterResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.pipeline import RetrieverPipeline
from backend.app.retrievers.pipeline.steps.document_routing_step import DocumentRoutingStep
from backend.app.retrievers.pipeline.steps.retrieval_planning_step import (
    RetrievalPlanningStep,
)
from backend.app.retrievers.query_understanding import QueryUnderstandingResult


def test_document_identity_candidate_routes_to_single_document():
    result = DocumentRouter().route(
        understanding=_understanding(document_hints=["product guide"]),
        rewrite_result=_rewrite(),
        metadata_filter_result=_filter_result([42]),
        max_candidates=20,
    )

    assert result.route_type == "DOCUMENT"
    assert result.target_document_ids == [42]
    assert result.reason == "document_identity_match"


def test_document_hints_route_to_document_candidates():
    result = DocumentRouter().route(
        understanding=_understanding(document_hints=["deployment manual"]),
        rewrite_result=_rewrite(),
        metadata_filter_result=_filter_result([10, 11]),
        max_candidates=20,
    )

    assert result.route_type == "DOCUMENT"
    assert result.target_document_ids == [10, 11]
    assert result.candidate_document_ids == [10, 11]


def test_single_document_route_from_metadata_filter():
    context = RetrieverPipelineContext(query="manual summary")
    context.query_understanding = _understanding(document_hints=["manual"])
    context.query_rewrite_result = _rewrite()
    context.auto_filter_result = _filter_result([7])

    result = DocumentRoutingStep().run(context)

    assert result.routing_result is not None
    assert result.routing_result.route_type == "DOCUMENT"
    assert result.metadata["document_routing"]["target_document_ids"] == [7]


def test_multi_document_route_for_comparison_query():
    result = DocumentRouter().route(
        understanding=_understanding(
            intent="comparison",
            document_hints=["policy a", "policy b"],
            comparison_targets=["policy a", "policy b"],
        ),
        rewrite_result=_rewrite(),
        metadata_filter_result=_filter_result([1, 2]),
        max_candidates=20,
    )

    assert result.route_type == "MULTI_DOCUMENT"
    assert result.target_document_ids == [1, 2]


def test_knowledge_base_route_when_no_document_scope_detected():
    result = DocumentRouter().route(
        understanding=_understanding(),
        rewrite_result=_rewrite(),
        metadata_filter_result=_filter_result([]),
        max_candidates=20,
    )

    assert result.route_type == "KNOWLEDGE_BASE"
    assert result.target_document_ids == []


def test_unknown_route_when_information_is_insufficient():
    result = DocumentRouter().route(
        understanding=QueryUnderstandingResult(
            original_query="",
            normalized_query="",
            intent="open_query",
        ),
        rewrite_result=_rewrite(""),
        metadata_filter_result=_filter_result([]),
        max_candidates=20,
    )

    assert result.route_type == "UNKNOWN"
    assert result.reason == "insufficient_query_information"


def test_document_router_fail_open_to_knowledge_base(monkeypatch):
    class BrokenRouter:
        def route(self, **kwargs):
            raise RuntimeError("router broken")

    monkeypatch.setattr(settings, "DOCUMENT_ROUTER_FAIL_OPEN", True)
    context = RetrieverPipelineContext(query="any query")

    result = DocumentRoutingStep(router=BrokenRouter()).run(context)

    assert result.routing_result is not None
    assert result.routing_result.route_type == "KNOWLEDGE_BASE"
    assert result.metadata["document_routing"]["failed"] is True
    assert result.metadata["document_routing"]["error"] == "router broken"


def test_retrieval_planning_reuses_routing_result():
    context = RetrieverPipelineContext(query="manual summary")
    context.query_understanding = _understanding(document_hints=["manual"])
    context.auto_filter_result = _filter_result([100])
    context = DocumentRoutingStep().run(context)

    result = RetrievalPlanningStep().run(context)

    assert result.retrieval_plan is not None
    assert result.retrieval_plan.document_ids == [100]
    assert result.metadata["retrieval_planning"]["document_routing_reused"] is True
    assert result.metadata["retrieval_planning"]["document_ids"] == [100]


def test_pipeline_order_has_document_router_after_metadata_filter_before_planning():
    pipeline = RetrieverPipeline()

    step_names = [step.__class__.__name__ for step in pipeline.steps]

    assert step_names.index("MetadataFilterStep") < step_names.index("DocumentRoutingStep")
    assert step_names.index("DocumentRoutingStep") < step_names.index(
        "RetrievalPlanningStep"
    )


def test_router_limits_candidate_count():
    result = DocumentRouter().route(
        understanding=_understanding(document_hints=["manual"]),
        rewrite_result=_rewrite(),
        metadata_filter_result=_filter_result([1, 2, 3]),
        max_candidates=2,
    )

    assert result.candidate_document_ids == [1, 2]
    assert result.target_document_ids == [1, 2]


def _understanding(
    *,
    intent: str = "open_query",
    document_hints: list[str] | None = None,
    comparison_targets: list[str] | None = None,
) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        original_query="generic query",
        normalized_query="generic query",
        intent=intent,
        document_hints=document_hints or [],
        comparison_targets=comparison_targets or [],
        confidence=0.8,
    )


def _rewrite(query: str = "generic query") -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=query,
        rewritten_query=query,
        rewrite_type="NONE",
        rewrite_changed=False,
        changed=False,
    )


def _filter_result(candidate_ids: list[int]) -> AutoMetadataFilterResult:
    return AutoMetadataFilterResult(
        candidate_document_ids=candidate_ids,
        auto_filter_applied=bool(candidate_ids),
        metadata={
            "strategy": "document_identity_keyword_match",
            "matched_documents": [
                {"document_id": document_id} for document_id in candidate_ids
            ],
        },
    )
