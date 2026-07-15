from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.document_router import RoutingResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.pipeline import RetrieverPipeline
from backend.app.retrievers.pipeline.steps.dense_retrieve_step import DenseRetrieveStep
from backend.app.retrievers.pipeline.steps.sparse_retrieve_step import SparseRetrieveStep
from backend.app.retrievers.pipeline.steps.strategy_selection_step import (
    StrategySelectionStep,
)
from backend.app.retrievers.planning import RetrievalPlan
from backend.app.retrievers.strategy import RetrievalStrategy, StrategySelector


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        document_constraint = next(
            (
                constraint
                for constraint in query.constraints
                if constraint.field == "document_id"
            ),
            None,
        )
        if document_constraint is None:
            return [_chunk(1), _chunk(2), _chunk(2), _chunk(3)]
        document_id = document_constraint.value
        return [
            _chunk(document_id, chunk_index=0),
            _chunk(document_id, chunk_index=1),
            _chunk(document_id, chunk_index=2),
        ]


def test_document_strategy_maps_to_document_scope():
    strategy = StrategySelector().select(
        routing_result=RoutingResult(
            route_type="DOCUMENT",
            target_document_ids=[10],
            candidate_document_ids=[10],
        ),
        requested_top_k=8,
        max_top_k=20,
        per_document_top_k=5,
    )

    assert strategy.strategy == "DOCUMENT"
    assert strategy.document_ids == [10]
    assert strategy.global_budget == 8
    assert strategy.per_document_budget == 8
    assert strategy.fallback is False


def test_multi_document_strategy_sets_per_document_budget():
    strategy = StrategySelector().select(
        routing_result=RoutingResult(
            route_type="MULTI_DOCUMENT",
            target_document_ids=[1, 2, 3, 4],
            candidate_document_ids=[1, 2, 3, 4],
        ),
        requested_top_k=20,
        max_top_k=20,
        per_document_top_k=5,
    )

    assert strategy.strategy == "MULTI_DOCUMENT"
    assert strategy.document_ids == [1, 2, 3, 4]
    assert strategy.global_budget == 20
    assert strategy.per_document_budget == 5


def test_knowledge_base_route_maps_to_global_strategy():
    strategy = StrategySelector().select(
        routing_result=RoutingResult(route_type="KNOWLEDGE_BASE"),
        requested_top_k=12,
        max_top_k=20,
        per_document_top_k=5,
    )

    assert strategy.strategy == "GLOBAL"
    assert strategy.document_ids == []
    assert strategy.fallback is False


def test_unknown_route_falls_back_to_global_strategy():
    strategy = StrategySelector().select(
        routing_result=RoutingResult(route_type="UNKNOWN"),
        requested_top_k=12,
        max_top_k=20,
        per_document_top_k=5,
    )

    assert strategy.strategy == "GLOBAL"
    assert strategy.fallback is True


def test_strategy_selection_step_writes_trace_metadata():
    context = RetrieverPipelineContext(query="compare docs", top_k=20)
    context.routing_result = RoutingResult(
        route_type="MULTI_DOCUMENT",
        target_document_ids=[1, 2, 3, 4],
    )

    result = StrategySelectionStep().run(context)

    assert result.retrieval_strategy is not None
    assert result.metadata["retrieval_strategy"]["strategy"] == "MULTI_DOCUMENT"
    assert result.metadata["retrieval_strategy"]["document_count"] == 4
    assert result.metadata["retrieval_strategy"]["per_document_budget"] == 5
    assert result.metadata["retrieval_strategy"]["global_budget"] == 20


def test_dense_retrieve_consumes_multi_document_strategy_budget():
    retriever = RecordingRetriever()
    context = _strategy_context()

    result = DenseRetrieveStep(dense_retriever=retriever).run(context)

    assert len(retriever.queries) == 2
    assert [query.top_k for query in retriever.queries] == [2, 2]
    assert [chunk.document_id for chunk in result.dense_chunks] == [1, 1, 2, 2]


def test_sparse_retrieve_consumes_multi_document_strategy_budget():
    retriever = RecordingRetriever()
    context = _strategy_context()

    result = SparseRetrieveStep(sparse_retriever=retriever).run(context)

    assert len(retriever.queries) == 2
    assert [query.top_k for query in retriever.queries] == [2, 2]
    assert [chunk.document_id for chunk in result.sparse_chunks] == [1, 1, 2, 2]


def test_pipeline_order_has_strategy_after_planning_before_retrieval():
    pipeline = RetrieverPipeline()

    step_names = [step.__class__.__name__ for step in pipeline.steps]

    assert step_names.index("RetrievalPlanningStep") < step_names.index(
        "StrategySelectionStep"
    )
    assert step_names.index("StrategySelectionStep") < step_names.index(
        "DenseRetrieveStep"
    )
    assert step_names.index("StrategySelectionStep") < step_names.index(
        "SparseRetrieveStep"
    )


def _strategy_context() -> RetrieverPipelineContext:
    context = RetrieverPipelineContext(query="compare docs", top_k=4)
    context.retrieval_plan = RetrievalPlan(
        original_query="compare docs",
        rewritten_query="compare docs",
        intent="hybrid",
        strategy="hybrid",
        document_ids=[1, 2],
        dense_enabled=True,
        sparse_enabled=True,
    )
    context.retrieval_strategy = RetrievalStrategy(
        strategy="MULTI_DOCUMENT",
        route_type="MULTI_DOCUMENT",
        document_ids=[1, 2],
        per_document_budget=2,
        global_budget=4,
    )
    return context


def _chunk(document_id: int, chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}_{chunk_index}",
        score=1.0,
        text=f"doc {document_id} chunk {chunk_index}",
        document_id=document_id,
        knowledge_base_id=1,
        chunk_index=chunk_index,
        metadata={},
    )
