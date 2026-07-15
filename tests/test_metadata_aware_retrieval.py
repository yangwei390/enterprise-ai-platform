from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.dense_retrieve_step import DenseRetrieveStep
from backend.app.retrievers.pipeline.steps.retrieval_planning_step import RetrievalPlanningStep
from backend.app.retrievers.pipeline.steps.sparse_retrieve_step import SparseRetrieveStep
from backend.app.retrievers.planning import (
    ConstraintEngine,
    FastQueryAnalyzer,
    RetrievalConstraint,
    RetrievalPlanner,
    RetrievalPlanningFactory,
    default_constraint_registry,
)
from backend.app.retrievers.sparse import BM25Index, SparseDocument, SparseSearchQuery


class ScopedRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        constraints = getattr(query, "constraints", [])
        if not constraints:
            return self.chunks
        engine = RetrievalPlanningFactory.get_constraint_engine()
        return [
            chunk
            for chunk in self.chunks
            if engine.matches_metadata(
                chunk.metadata,
                constraints,
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
            )
        ]


def test_chapter_query_builds_metadata_constraint():
    analysis = FastQueryAnalyzer(default_constraint_registry()).analyze("劳动法第二章讲什么")

    constraint = next(item for item in analysis.constraints if item.field == "chapter_number")
    assert constraint.operator == "eq"
    assert constraint.value == 2


def test_article_query_builds_metadata_constraint():
    analysis = FastQueryAnalyzer(default_constraint_registry()).analyze("劳动法第十条是什么")

    constraint = next(item for item in analysis.constraints if item.field == "article_number")
    assert constraint.operator == "eq"
    assert constraint.value == 10


def test_heading_query_builds_generic_heading_constraint():
    analysis = FastQueryAnalyzer(default_constraint_registry()).analyze("员工手册薪资制度")

    values = [constraint.value for constraint in analysis.constraints]
    assert "薪资制度" in values
    assert any(constraint.field == "heading_title" for constraint in analysis.constraints)


def test_no_metadata_query_keeps_normal_hybrid_plan():
    context = RetrieverPipelineContext(query="介绍一下平台能力")

    result = RetrievalPlanningStep().run(context)

    assert result.retrieval_plan is not None
    assert result.retrieval_plan.constraints == []
    assert result.retrieval_plan.strategy == "hybrid"


def test_dense_constraint_miss_falls_back_to_knowledge_base():
    retriever = ScopedRetriever([_chunk(1, chapter_number=3)])
    context = _planned_context("第二章")
    context.metadata["retrieval_planning"] = {}

    result = DenseRetrieveStep(dense_retriever=retriever).run(context)

    assert [chunk.metadata["chapter_number"] for chunk in result.dense_chunks] == [3]
    assert len(retriever.queries) == 2
    assert retriever.queries[0].constraints
    assert retriever.queries[1].constraints == []
    assert result.metadata["constraint_scope"]["dense_fallback_used"] is True


def test_sparse_constraint_scope_runs_before_bm25_top_k():
    index = BM25Index()
    index.add_documents(
        [
            SparseDocument(
                id="1",
                text="薪资制度 内容",
                document_id=1,
                knowledge_base_id=4,
                metadata={"heading_title": "安装说明"},
            ),
            SparseDocument(
                id="2",
                text="薪资制度 内容",
                document_id=2,
                knowledge_base_id=4,
                metadata={"heading_title": "薪资制度"},
            ),
        ]
    )

    results = index.search(
        SparseSearchQuery(
            query="薪资制度",
            top_k=1,
            constraints=[
                RetrievalConstraint(
                    field="heading_title",
                    operator="contains",
                    value="薪资",
                    applied=True,
                )
            ],
        )
    )

    assert [result.document_id for result in results] == [2]


def test_multi_document_retrieval_respects_document_id_constraint():
    retriever = ScopedRetriever(
        [
            _chunk(1, heading_title="薪资制度"),
            _chunk(2, heading_title="薪资制度"),
        ]
    )
    context = RetrieverPipelineContext(query="薪资制度", knowledge_base_id=4)
    context.metadata["retrieval_planning"] = {}
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="薪资制度",
        rewritten_query="薪资制度",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("薪资制度"),
    )
    context.retrieval_plan.constraints = [
        RetrievalConstraint(
            field="document_id",
            operator="in",
            value=[2],
            applied=True,
            source="test",
        )
    ]
    context.retrieval_plan.use_structure_filter = True

    result = SparseRetrieveStep(sparse_retriever=retriever).run(context)

    assert [chunk.document_id for chunk in result.sparse_chunks] == [2]


def test_constraint_engine_supports_prefix_and_range():
    engine = ConstraintEngine(default_constraint_registry())

    assert engine.matches_metadata(
        {"section_path": ["员工手册", "薪资制度"], "page_start": 5},
        [
            RetrievalConstraint(
                field="section_path",
                operator="prefix",
                value="员工",
                applied=True,
            ),
            RetrievalConstraint(
                field="page_start",
                operator="range",
                value={"gte": 3, "lte": 8},
                applied=True,
            ),
        ],
    )


def test_planning_metadata_logs_constraint_scope():
    context = RetrieverPipelineContext(query="劳动法第三节", knowledge_base_id=4)

    result = RetrievalPlanningStep().run(context)

    metadata = result.metadata["retrieval_planning"]
    assert metadata["query"] == "劳动法第三节"
    assert metadata["dense_scope"] == "metadata_constraints"
    assert metadata["sparse_scope"] == "metadata_constraints"
    assert metadata["planning_duration_ms"] >= 0


def _planned_context(query: str) -> RetrieverPipelineContext:
    context = RetrieverPipelineContext(query=query, knowledge_base_id=4)
    analysis = FastQueryAnalyzer(default_constraint_registry()).analyze(query)
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query=query,
        rewritten_query=query,
        candidate_document_ids=[],
        analysis=analysis,
    )
    return context


def _chunk(document_id: int, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}_{metadata.get('chunk_index', 0)}",
        score=1.0,
        text=metadata.get("text", "content"),
        document_id=document_id,
        knowledge_base_id=4,
        chunk_index=metadata.get("chunk_index", 0),
        metadata=metadata,
    )
