from backend.app.config.settings import settings
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.query_understanding_step import (
    QueryUnderstandingStep,
)
from backend.app.retrievers.pipeline.steps.retrieval_planning_step import (
    RetrievalPlanningStep,
)
from backend.app.retrievers.planning import FastQueryAnalyzer, default_constraint_registry
from backend.app.retrievers.query_understanding import (
    FastQueryAnalyzer as FastUnderstandingAnalyzer,
)
from backend.app.retrievers.query_understanding import QueryUnderstandingResult


def test_understands_factual_query():
    result = FastUnderstandingAnalyzer().analyze("企业知识库是什么")

    assert result.intent == "factual"
    assert result.normalized_query == "企业知识库是什么"
    assert result.confidence > 0


def test_understands_chapter_query():
    result = FastUnderstandingAnalyzer().analyze("制度第二章讲什么")

    assert result.intent == "structured"
    assert {"type": "chapter", "number": 2, "raw": "第二章", "source": "rule"} in (
        result.structure_hints
    )


def test_understands_article_query():
    result = FastUnderstandingAnalyzer().analyze("第十条是什么")

    assert result.intent == "structured"
    assert any(
        hint["type"] == "article" and hint["number"] == 10
        for hint in result.structure_hints
    )


def test_understands_summary_query():
    result = FastUnderstandingAnalyzer().analyze("概括员工手册主要内容")

    assert result.intent == "summary"
    assert "员工手册" in result.document_hints


def test_understands_multi_document_comparison_query():
    result = FastUnderstandingAnalyzer().analyze("对比员工手册和安装指南的不同点")

    assert result.intent == "multi_document"
    assert "员工手册" in result.document_hints
    assert "安装指南" in result.document_hints
    assert result.comparison_targets


def test_understands_exact_error_code_query():
    result = FastUnderstandingAnalyzer().analyze("ERR-1001 是什么原因")

    assert result.intent == "lexical"
    assert "ERR-1001" in result.metadata["exact_tokens"]
    assert "ERR-1001" in result.keywords


def test_understands_temporal_range_query():
    result = FastUnderstandingAnalyzer().analyze("查看2023年至2025年的费用制度变化")

    assert {"type": "year_range", "start": 2023, "end": 2025, "raw": "2023年至2025"} in (
        result.temporal_constraints
    )


def test_understands_negative_constraints():
    result = FastUnderstandingAnalyzer().analyze("总结安装指南，不要附件，排除旧版本")

    assert "附件" in result.negative_constraints
    assert "旧版本" in result.negative_constraints


def test_understands_empty_query_without_error():
    result = FastUnderstandingAnalyzer().analyze("")

    assert result.intent == "open_query"
    assert result.keywords == []


def test_query_understanding_step_fail_open(monkeypatch):
    class BrokenAnalyzer:
        def analyze(self, query: str):
            raise RuntimeError("broken analyzer")

    monkeypatch.setattr(settings, "QUERY_UNDERSTANDING_FAIL_OPEN", True)
    context = RetrieverPipelineContext(query="任意问题")

    result = QueryUnderstandingStep(analyzer=BrokenAnalyzer()).run(context)

    assert result.query_understanding is None
    assert result.metadata["query_understanding"]["failed"] is True
    assert result.metadata["query_understanding"]["error"] == "broken analyzer"


def test_retrieval_planner_reuses_query_understanding_result():
    understanding = QueryUnderstandingResult(
        original_query="制度第二章",
        normalized_query="制度第二章",
        intent="structured",
        structure_hints=[
            {"type": "chapter", "number": 2, "raw": "第二章", "source": "rule"}
        ],
        confidence=0.9,
    )

    analysis = FastQueryAnalyzer(default_constraint_registry()).analyze(
        query="这部分讲什么",
        rewritten_query="这部分讲什么",
        understanding=understanding,
    )

    assert analysis.metadata["analyzer"] == "query_understanding"
    assert any(
        constraint.field == "chapter_number" and constraint.value == 2
        for constraint in analysis.constraints
    )


def test_pipeline_metadata_includes_query_understanding():
    context = RetrieverPipelineContext(query="第十条是什么")

    understood = QueryUnderstandingStep().run(context)
    planned = RetrievalPlanningStep().run(understood)

    assert planned.metadata["query_understanding"]["enabled"] is True
    assert planned.metadata["query_understanding"]["intent"] == "structured"
    assert planned.retrieval_plan is not None
    assert planned.retrieval_plan.metadata["analysis"]["analyzer"] == "query_understanding"


def test_query_understanding_step_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_UNDERSTANDING_ENABLED", False)
    context = RetrieverPipelineContext(query="第十条是什么")

    result = QueryUnderstandingStep().run(context)

    assert result.query_understanding is None
    assert result.metadata["query_understanding"]["enabled"] is False
