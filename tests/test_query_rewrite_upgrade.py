from backend.app.config.settings import settings
from backend.app.query import SimpleQueryRewriter
from backend.app.retrievers.hybrid import HybridRetriever
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.pipeline import RetrieverPipeline
from backend.app.retrievers.pipeline.steps.query_rewrite_step import QueryRewriteStep
from backend.app.retrievers.pipeline.steps.query_understanding_step import (
    QueryUnderstandingStep,
)
from backend.app.retrievers.query_understanding import (
    FastQueryAnalyzer,
    QueryUnderstandingResult,
)


def test_rewrites_normal_question_with_normalization():
    understanding = FastQueryAnalyzer().analyze("请问企业知识库是什么？")

    result = SimpleQueryRewriter().rewrite("请问企业知识库是什么？", understanding)

    assert result.rewrite_type == "NORMALIZATION"
    assert result.rewrite_changed is True
    assert result.rewritten_query == "企业知识库"


def test_pipeline_runs_query_understanding_before_query_rewrite():
    pipeline = RetrieverPipeline()

    step_names = [step.__class__.__name__ for step in pipeline.steps]

    assert step_names.index("QueryUnderstandingStep") < step_names.index("QueryRewriteStep")


def test_hybrid_pipeline_runs_query_understanding_before_query_rewrite():
    retriever = HybridRetriever()

    step_names = [step.__class__.__name__ for step in retriever.pipeline.steps]

    assert step_names.index("QueryUnderstandingStep") < step_names.index("QueryRewriteStep")


def test_query_rewrite_step_does_not_create_query_understanding():
    context = RetrieverPipelineContext(query="帮我看看员工手册主要内容")

    result = QueryRewriteStep().run(context)

    assert result.query_understanding is None
    assert result.metadata["query_rewrite"]["metadata"]["understanding_used"] is False
    assert result.metadata["query_rewrite"]["after"] == "员工手册主要内容"


def test_query_rewrite_consumes_existing_query_understanding():
    context = RetrieverPipelineContext(query="帮我看看员工手册主要内容")
    context = QueryUnderstandingStep().run(context)
    assert context.query_understanding is not None

    result = QueryRewriteStep().run(context)

    assert result.query_understanding.document_hints == ["员工手册"]
    assert result.metadata["query_rewrite"]["metadata"]["understanding_used"] is True
    assert result.metadata["query_rewrite"]["after"] == "员工手册主要内容"


def test_summary_query_gets_light_expansion():
    understanding = FastQueryAnalyzer().analyze("总结员工手册")

    result = SimpleQueryRewriter().rewrite("总结员工手册", understanding)

    assert result.rewrite_type == "EXPANSION"
    assert result.rewritten_query == "总结员工手册 主要内容"


def test_structured_query_only_normalizes():
    understanding = FastQueryAnalyzer().analyze("请问劳动法第十条讲什么")

    result = SimpleQueryRewriter().rewrite("请问劳动法第十条讲什么", understanding)

    assert result.rewrite_type == "NORMALIZATION"
    assert result.rewritten_query == "劳动法 第十条"
    assert "主要内容" not in result.rewritten_query
    assert understanding.structure_hints[0]["raw"] == "第十条"


def test_chapter_query_preserves_structure_hint():
    context = RetrieverPipelineContext(query="请问制度第二章讲什么")

    understood = QueryUnderstandingStep().run(context)
    rewritten = QueryRewriteStep().run(understood)

    assert rewritten.query_understanding is not None
    assert rewritten.query_understanding.structure_hints[0]["number"] == 2
    assert rewritten.metadata["query_rewrite"]["after"] == "制度 第二章"


def test_error_code_query_is_not_rewritten():
    understanding = FastQueryAnalyzer().analyze("ERR-1001 是什么")

    result = SimpleQueryRewriter().rewrite("ERR-1001 是什么", understanding)

    assert result.rewrite_type == "NONE"
    assert result.rewritten_query == "ERR-1001 是什么"


def test_version_query_is_not_rewritten():
    understanding = FastQueryAnalyzer().analyze("v2.3 有什么变化")

    result = SimpleQueryRewriter().rewrite("v2.3 有什么变化", understanding)

    assert result.rewrite_type == "NONE"
    assert result.rewritten_query == "v2.3 有什么变化"


def test_filename_query_is_not_rewritten():
    understanding = FastQueryAnalyzer().analyze("abc.pdf 主要内容")

    result = SimpleQueryRewriter().rewrite("abc.pdf 主要内容", understanding)

    assert result.rewrite_type == "NONE"
    assert result.rewritten_query == "abc.pdf 主要内容"


def test_document_identity_hint_is_preserved_without_domain_hardcoding():
    understanding = FastQueryAnalyzer().analyze("劳动合同")

    result = SimpleQueryRewriter().rewrite("劳动合同", understanding)

    assert result.rewrite_type == "NONE"
    assert result.rewritten_query == "劳动合同"
    assert understanding.document_hints == ["劳动合同"]


def test_rewrite_has_no_labor_law_or_employment_domain_hardcoding():
    for query in ("劳动法", "劳动合同", "促进就业"):
        understanding = QueryUnderstandingResult(
            original_query=query,
            normalized_query=query,
            intent="open_query",
            document_hints=[query],
            confidence=0.8,
        )

        result = SimpleQueryRewriter().rewrite(query, understanding)

        assert result.rewritten_query == query
        assert result.rewrite_type == "NONE"
        assert "相关规定" not in result.rewritten_query
        assert "主要内容" not in result.rewritten_query


def test_comparison_query_adds_comparison_keyword_without_changing_targets():
    understanding = FastQueryAnalyzer().analyze("对比员工手册和安装指南")

    result = SimpleQueryRewriter().rewrite("对比员工手册和安装指南", understanding)

    assert result.rewrite_type == "KEYWORD_ENRICHMENT"
    assert result.rewritten_query == "对比员工手册和安装指南 区别"
    assert understanding.document_hints == ["员工手册", "安装指南"]


def test_lexical_query_is_not_rewritten():
    understanding = FastQueryAnalyzer().analyze("HTTP404")

    result = SimpleQueryRewriter().rewrite("HTTP404", understanding)

    assert result.rewrite_type == "NONE"
    assert result.rewritten_query == "HTTP404"


def test_rewrite_fail_open(monkeypatch):
    class BrokenRewriter:
        def rewrite(self, query, understanding=None, max_length=None):
            raise RuntimeError("rewrite broken")

    monkeypatch.setattr(settings, "QUERY_REWRITE_FAIL_OPEN", True)
    context = RetrieverPipelineContext(query="任意问题")

    result = QueryRewriteStep(rewriter=BrokenRewriter()).run(context)

    assert result.rewritten_query == "任意问题"
    assert result.metadata["query_rewrite"]["failed"] is True
    assert result.metadata["query_rewrite"]["error"] == "rewrite broken"


def test_rewrite_disabled(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)
    context = RetrieverPipelineContext(query="请问企业知识库是什么")

    result = QueryRewriteStep().run(context)

    assert result.rewritten_query == "请问企业知识库是什么"
    assert result.metadata["query_rewrite"]["enabled"] is False
