from typing import Any, cast

from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps import (
    DenseRetrieveStep,
    DocumentRoutingStep,
    FusionStep,
    MetadataFilterStep,
    SparseRetrieveStep,
    StrategySelectionStep,
)
from backend.app.retrievers.planning import RetrievalConstraint, RetrievalPlan
from backend.app.retrievers.qdrant_retriever import QdrantRetriever
from backend.app.retrievers.sparse import BM25Index, SparseDocument, SparseSearchQuery


class RecordingRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        constraints = getattr(query, "constraints", [])
        document_constraints = [
            constraint
            for constraint in constraints
            if constraint.field == "document_id" and constraint.operator == "eq"
        ]
        if not document_constraints:
            return self.chunks
        allowed_id = document_constraints[-1].value
        return [chunk for chunk in self.chunks if chunk.document_id == allowed_id]


class EmptyThenRecordingRetriever(RecordingRetriever):
    def retrieve(self, query):
        self.queries.append(query)
        if len(self.queries) == 1:
            return []
        constraints = getattr(query, "constraints", [])
        document_constraints = [
            constraint
            for constraint in constraints
            if constraint.field == "document_id" and constraint.operator == "eq"
        ]
        if not document_constraints:
            return self.chunks
        allowed_id = document_constraints[-1].value
        return [chunk for chunk in self.chunks if chunk.document_id == allowed_id]


def test_explicit_document_id_builds_document_route_without_database_lookup() -> None:
    context = RetrieverPipelineContext(
        query="这款豆浆机怎么清洗",
        knowledge_base_id=4,
        document_id=12,
    )

    context = MetadataFilterStep().run(context)
    context = DocumentRoutingStep().run(context)
    context = StrategySelectionStep().run(context)

    assert context.auto_filter_result is not None
    assert context.auto_filter_result.candidate_document_ids == [12]
    assert context.auto_filter_result.metadata["strategy"] == "explicit_document_id"
    assert context.routing_result is not None
    assert context.routing_result.route_type == "DOCUMENT"
    assert context.routing_result.target_document_ids == [12]
    assert context.retrieval_strategy is not None
    assert context.retrieval_strategy.strategy == "DOCUMENT"
    assert context.retrieval_strategy.document_ids == [12]


def test_dense_and_sparse_use_same_document_id_constraint_for_document_route() -> None:
    context = RetrieverPipelineContext(
        query="这款豆浆机怎么清洗",
        knowledge_base_id=4,
        document_id=12,
    )
    context = MetadataFilterStep().run(context)
    context = DocumentRoutingStep().run(context)
    context = StrategySelectionStep().run(context)

    chunks = [_chunk(9), _chunk(12)]
    dense_retriever = RecordingRetriever(chunks)
    sparse_retriever = RecordingRetriever(chunks)

    context = DenseRetrieveStep(dense_retriever=cast(Any, dense_retriever)).run(context)
    context = SparseRetrieveStep(sparse_retriever=cast(Any, sparse_retriever)).run(context)

    assert [chunk.document_id for chunk in context.dense_chunks] == [12]
    assert [chunk.document_id for chunk in context.sparse_chunks] == [12]
    assert dense_retriever.queries[0].constraints[-1].field == "document_id"
    assert dense_retriever.queries[0].constraints[-1].value == 12
    assert sparse_retriever.queries[0].constraints[-1].field == "document_id"
    assert sparse_retriever.queries[0].constraints[-1].value == 12


def test_dense_and_sparse_apply_document_id_constraint_without_router_strategy() -> None:
    context = RetrieverPipelineContext(
        query="这款豆浆机怎么清洗",
        knowledge_base_id=4,
        document_id=12,
    )
    chunks = [_chunk(9), _chunk(12)]
    dense_retriever = RecordingRetriever(chunks)
    sparse_retriever = RecordingRetriever(chunks)

    context = DenseRetrieveStep(dense_retriever=cast(Any, dense_retriever)).run(context)
    context = SparseRetrieveStep(sparse_retriever=cast(Any, sparse_retriever)).run(context)

    assert [chunk.document_id for chunk in context.dense_chunks] == [12]
    assert [chunk.document_id for chunk in context.sparse_chunks] == [12]
    assert dense_retriever.queries[0].constraints[-1].source == "explicit_document_filter"
    assert sparse_retriever.queries[0].constraints[-1].source == "explicit_document_filter"


def test_structured_fallback_keeps_explicit_document_id_constraint() -> None:
    context = RetrieverPipelineContext(
        query="第二章怎么清洗",
        knowledge_base_id=4,
        document_id=12,
    )
    context.retrieval_plan = RetrievalPlan(
        original_query=context.query,
        rewritten_query=context.query,
        intent="structured",
        strategy="hybrid",
        constraints=[
            RetrievalConstraint(
                field="chapter_number",
                operator="eq",
                value=2,
                applied=True,
            )
        ],
        use_structure_filter=True,
    )
    retriever = EmptyThenRecordingRetriever([_chunk(9), _chunk(12)])

    result = DenseRetrieveStep(dense_retriever=cast(Any, retriever)).run(context)

    assert [chunk.document_id for chunk in result.dense_chunks] == [12]
    assert len(retriever.queries) == 2
    fallback_constraints = retriever.queries[1].constraints
    assert [constraint.field for constraint in fallback_constraints] == ["document_id"]
    assert fallback_constraints[0].source == "explicit_document_filter"


def test_fusion_keeps_only_explicit_document_id_chunks() -> None:
    context = RetrieverPipelineContext(
        query="这款豆浆机怎么清洗",
        knowledge_base_id=4,
        document_id=12,
    )
    context = MetadataFilterStep().run(context)
    context.dense_chunks = [_chunk(9), _chunk(12)]
    context.sparse_chunks = [_chunk(9, chunk_index=1), _chunk(12, chunk_index=1)]

    result = FusionStep().run(context)

    assert {chunk.document_id for chunk in result.fused_chunks} == {12}
    assert result.metadata["retrieval_scope"]["fusion_scope_guard_applied"] is True


def test_qdrant_document_id_constraint_targets_top_level_payload() -> None:
    constraint = RetrievalConstraint(
        field="document_id",
        operator="eq",
        value=12,
        applied=True,
    )

    query_filter = QdrantRetriever()._build_filter(
        knowledge_base_id=4,
        metadata_filter=None,
        constraints=[constraint],
    )

    assert query_filter is not None
    payload = query_filter.model_dump(mode="json", exclude_none=True)
    keys = [item["key"] for item in payload["must"]]
    assert "knowledge_base_id" in keys
    assert "document_id" in keys
    assert "metadata.document_id" not in keys


def test_bm25_document_id_constraint_filters_top_level_document_id() -> None:
    index = BM25Index()
    index.add_documents(
        [
            SparseDocument(
                id="9_0",
                text="豆浆机 清洗",
                document_id=9,
                knowledge_base_id=4,
                chunk_index=0,
            ),
            SparseDocument(
                id="12_0",
                text="豆浆机 清洗",
                document_id=12,
                knowledge_base_id=4,
                chunk_index=0,
            ),
        ]
    )

    results = index.search(
        SparseSearchQuery(
            query="豆浆机 清洗",
            knowledge_base_id=4,
            top_k=10,
            constraints=[
                RetrievalConstraint(
                    field="document_id",
                    operator="eq",
                    value=12,
                    applied=True,
                )
            ],
        )
    )

    assert [result.document_id for result in results] == [12]


def _chunk(document_id: int, chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}_{chunk_index}",
        score=1.0,
        text=f"doc {document_id}",
        document_id=document_id,
        knowledge_base_id=4,
        chunk_index=chunk_index,
        metadata={"source": f"manual-{document_id}.pdf"},
    )
