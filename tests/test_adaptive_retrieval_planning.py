from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.metadata_filter import AutoMetadataFilterResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.dense_retrieve_step import DenseRetrieveStep
from backend.app.retrievers.pipeline.steps.fusion_step import FusionStep
from backend.app.retrievers.pipeline.steps.retrieval_planning_step import RetrievalPlanningStep
from backend.app.retrievers.pipeline.steps.soft_boost_step import SoftBoostStep
from backend.app.retrievers.pipeline.steps.sparse_retrieve_step import SparseRetrieveStep
from backend.app.retrievers.planning import (
    ConstraintEngine,
    ConstraintFieldDefinition,
    ConstraintRegistry,
    FastQueryAnalyzer,
    RetrievalConstraint,
    RetrievalPlanner,
    RetrievalPlanningFactory,
    default_constraint_registry,
)
from backend.app.retrievers.sparse import BM25Index, SparseDocument, SparseSearchQuery


def _chunk(document_id: int, score: float = 1.0, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}_{metadata.get('chunk_index', 0)}",
        score=score,
        text=metadata.get("text", "content"),
        document_id=document_id,
        knowledge_base_id=4,
        chunk_index=metadata.get("chunk_index", 0),
        metadata=metadata,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_query = None

    def retrieve(self, query):
        self.last_query = query
        constraints = getattr(query, "constraints", [])
        if constraints:
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
        return self.chunks


def test_constraint_registry_registers_fields() -> None:
    registry = default_constraint_registry()

    assert registry.get("chapter_number") is not None
    assert registry.get("章").field == "chapter_number"


def test_registry_rejects_duplicate_invalid_field() -> None:
    registry = ConstraintRegistry()
    registry.register(
        ConstraintFieldDefinition(
            field="chapter_number",
            value_type="int",
            allowed_operators=["eq"],
        )
    )

    try:
        registry.register(
            ConstraintFieldDefinition(
                field="chapter_number",
                value_type="int",
                allowed_operators=["eq"],
            )
        )
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate field should fail")


def test_chapter_query_creates_generic_constraint() -> None:
    result = FastQueryAnalyzer(default_constraint_registry()).analyze("第二章讲什么")

    assert result.constraints[0].field == "chapter_number"
    assert result.constraints[0].value == 2


def test_article_query_creates_generic_constraint() -> None:
    result = FastQueryAnalyzer(default_constraint_registry()).analyze("第十条是什么")

    assert result.constraints[0].field == "article_number"
    assert result.constraints[0].value == 10


def test_normal_semantic_query_has_no_structure_constraint() -> None:
    result = FastQueryAnalyzer(default_constraint_registry()).analyze("介绍一下平台能力")

    assert result.constraints == []


def test_error_code_query_detects_lexical_intent() -> None:
    result = FastQueryAnalyzer(default_constraint_registry()).analyze("LA-403 是什么错误")

    assert result.intent == "lexical"


def test_normal_query_selects_hybrid() -> None:
    registry = default_constraint_registry()
    analysis = FastQueryAnalyzer(registry).analyze("介绍一下平台能力")

    plan = RetrievalPlanner(registry).plan(
        query="介绍一下平台能力",
        rewritten_query="介绍一下平台能力",
        candidate_document_ids=[],
        analysis=analysis,
    )

    assert plan.strategy == "hybrid"


def test_structure_query_selects_structured_hybrid() -> None:
    registry = default_constraint_registry()
    analysis = FastQueryAnalyzer(registry).analyze("第二章讲什么")

    plan = RetrievalPlanner(registry).plan(
        query="第二章讲什么",
        rewritten_query="第二章讲什么",
        candidate_document_ids=[],
        analysis=analysis,
    )

    assert plan.strategy == "structured_hybrid"
    assert plan.use_structure_filter is True


def test_planner_does_not_call_llm_by_default() -> None:
    result = FastQueryAnalyzer(default_constraint_registry()).analyze("第二章讲什么")

    assert result.metadata["llm_called"] is False


def test_planning_fail_open_to_hybrid(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("planner failed")

    monkeypatch.setattr(FastQueryAnalyzer, "analyze", fail)
    context = RetrieverPipelineContext(query="x")

    result = RetrievalPlanningStep().run(context)

    assert result.retrieval_plan.strategy == "hybrid"
    assert result.retrieval_plan.fallback_used is True


def test_qdrant_constraint_eq_conversion() -> None:
    engine = ConstraintEngine(default_constraint_registry())
    query_filter = engine.to_qdrant_filter(
        [RetrievalConstraint(field="chapter_number", operator="eq", value=2, applied=True)]
    )

    dumped = query_filter.model_dump(mode="json", exclude_none=True)
    assert dumped["must"][0]["key"] == "metadata.chapter_number"
    assert dumped["must"][0]["match"]["value"] == 2


def test_qdrant_constraint_in_conversion() -> None:
    engine = ConstraintEngine(default_constraint_registry())
    query_filter = engine.to_qdrant_filter(
        [
            RetrievalConstraint(
                field="chapter_number",
                operator="in",
                value=[2, 3],
                applied=True,
            )
        ]
    )

    dumped = query_filter.model_dump(mode="json", exclude_none=True)
    assert dumped["must"][0]["match"]["any"] == [2, 3]


def test_bm25_constraint_eq_before_top_k() -> None:
    index = BM25Index()
    index.add_documents(
        [
            SparseDocument(
                id="1",
                text="第二章 就业",
                document_id=1,
                metadata={"chapter_number": 1},
            ),
            SparseDocument(
                id="2",
                text="第二章 就业",
                document_id=2,
                metadata={"chapter_number": 2},
            ),
        ]
    )

    results = index.search(
        SparseSearchQuery(
            query="第二章",
            top_k=5,
            constraints=[
                RetrievalConstraint(
                    field="chapter_number",
                    operator="eq",
                    value=2,
                    applied=True,
                )
            ],
        )
    )

    assert [result.document_id for result in results] == [2]


def test_bm25_constraint_contains() -> None:
    engine = ConstraintEngine(default_constraint_registry())

    assert engine.matches_metadata(
        {"heading_title": "Agent Runtime"},
        [
            RetrievalConstraint(
                field="heading_title",
                operator="contains",
                value="Agent",
                applied=True,
            )
        ],
    )


def test_unknown_field_rejected() -> None:
    planner = RetrievalPlanner(default_constraint_registry())
    accepted, rejected = planner._validate_constraints(
        [RetrievalConstraint(field="unknown", operator="eq", value=1)]
    )

    assert accepted == []
    assert rejected[0].rejected_reason == "unknown_field"


def test_invalid_operator_rejected() -> None:
    planner = RetrievalPlanner(default_constraint_registry())
    accepted, rejected = planner._validate_constraints(
        [RetrievalConstraint(field="chapter_number", operator="contains", value=1)]
    )

    assert accepted == []
    assert rejected[0].rejected_reason == "invalid_operator"


def test_type_mismatch_handled() -> None:
    planner = RetrievalPlanner(default_constraint_registry())
    accepted, rejected = planner._validate_constraints(
        [RetrievalConstraint(field="chapter_number", operator="eq", value="二")]
    )

    assert accepted == []
    assert rejected[0].rejected_reason == "type_mismatch"


def test_chapter_constraint_works_without_legal_specific_retriever() -> None:
    retriever = FakeRetriever([_chunk(12, chapter_number=2), _chunk(12, chapter_number=3)])
    context = RetrieverPipelineContext(query="第二章", knowledge_base_id=4)
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="第二章",
        rewritten_query="第二章",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("第二章"),
    )

    result = DenseRetrieveStep(dense_retriever=retriever).run(context)

    assert [chunk.metadata["chapter_number"] for chunk in result.dense_chunks] == [2]


def test_article_constraint_works_without_legal_specific_retriever() -> None:
    retriever = FakeRetriever([_chunk(12, article_number=10), _chunk(12, article_number=11)])
    context = RetrieverPipelineContext(query="第十条", knowledge_base_id=4)
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="第十条",
        rewritten_query="第十条",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("第十条"),
    )

    result = SparseRetrieveStep(sparse_retriever=retriever).run(context)

    assert [chunk.metadata["article_number"] for chunk in result.sparse_chunks] == [10]


def test_markdown_heading_constraint_can_be_registered_and_executed() -> None:
    engine = ConstraintEngine(default_constraint_registry())

    assert engine.matches_metadata(
        {"heading_title": "Agent"},
        [
            RetrievalConstraint(
                field="heading_title",
                operator="eq",
                value="Agent",
                applied=True,
            )
        ],
    )


def test_candidate_document_ids_remains_compatible() -> None:
    context = RetrieverPipelineContext(query="x")
    context.auto_filter_result = AutoMetadataFilterResult(candidate_document_ids=[12])

    result = RetrievalPlanningStep().run(context)

    assert result.retrieval_plan.document_ids == [12]


def test_dense_respects_generic_constraints() -> None:
    retriever = FakeRetriever([_chunk(12, chapter_number=2), _chunk(12, chapter_number=3)])
    context = RetrieverPipelineContext(query="第二章", knowledge_base_id=4)
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="第二章",
        rewritten_query="第二章",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("第二章"),
    )

    DenseRetrieveStep(dense_retriever=retriever).run(context)

    assert retriever.last_query.constraints[0].field == "chapter_number"


def test_sparse_respects_generic_constraints() -> None:
    retriever = FakeRetriever([_chunk(12, article_number=10), _chunk(12, article_number=11)])
    context = RetrieverPipelineContext(query="第十条", knowledge_base_id=4)
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="第十条",
        rewritten_query="第十条",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("第十条"),
    )

    SparseRetrieveStep(sparse_retriever=retriever).run(context)

    assert retriever.last_query.constraints[0].field == "article_number"


def test_fusion_receives_retrieval_plan() -> None:
    context = RetrieverPipelineContext(query="LA-403", knowledge_base_id=4)
    context.dense_chunks = [_chunk(1, 0.2)]
    context.sparse_chunks = [_chunk(2, 1.0)]
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="LA-403",
        rewritten_query="LA-403",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("LA-403"),
    )

    result = FusionStep().run(context)

    assert result.metadata["fusion_plan"]["intent"] == "lexical"
    assert result.metadata["fusion"] == "sparse_first"


def test_structure_no_match_fail_open_hybrid() -> None:
    retriever = FakeRetriever([_chunk(12, chapter_number=3)])
    context = RetrieverPipelineContext(query="第二章", knowledge_base_id=4)
    context.metadata["retrieval_planning"] = {}
    context.retrieval_plan = RetrievalPlanner(default_constraint_registry()).plan(
        query="第二章",
        rewritten_query="第二章",
        candidate_document_ids=[],
        analysis=FastQueryAnalyzer(default_constraint_registry()).analyze("第二章"),
    )

    result = DenseRetrieveStep(dense_retriever=retriever).run(context)

    assert [chunk.metadata["chapter_number"] for chunk in result.dense_chunks] == [3]
    assert result.retrieval_plan.fallback_used is True


def test_existing_structure_soft_boost_remains_compatible() -> None:
    context = RetrieverPipelineContext(query="第二章", knowledge_base_id=4)
    context.fused_chunks = [_chunk(12, chapter_number=2), _chunk(12, chapter_number=3)]

    result = SoftBoostStep().run(context)

    assert result.metadata["structure_soft_boost_applied"] is True


def test_chat_agent_workflow_runtime_configs_remain_compatible() -> None:
    from backend.app.config.settings import settings

    assert settings.AGENT_RUNTIME in {"v1", "langgraph"}
    assert settings.WORKFLOW_RUNTIME in {"v1", "langgraph"}
