import asyncio
from pathlib import Path
from typing import Any

import pytest
from backend.app.api.evaluation import router as evaluation_router
from backend.app.config.settings import settings
from backend.app.tools import BaseTool, ToolDescriptor, ToolResult
from backend.app.tools.registry import get_tool_registry
from evaluation.v2.baseline import BaselineManager
from evaluation.v2.errors import EvaluationBaselineError
from evaluation.v2.history import HistoryManager
from evaluation.v2.metrics import get_metric, list_metrics
from evaluation.v2.metrics.generation import KeywordCoverageMetric
from evaluation.v2.metrics.retrieval import ChunkRecallMetric, RetrieverHitMetric
from evaluation.v2.regression import RegressionComparator
from evaluation.v2.report import build_report, redact_and_truncate
from evaluation.v2.runner import EvaluationRunnerV2
from evaluation.v2.schemas import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationContext,
    EvaluationSuite,
    EvaluationTargetResult,
    MetricResult,
)
from evaluation.v2.suite import load_suite, load_v1_questions_as_suite
from evaluation.v2.targets.agent import AgentEvaluationTarget
from evaluation.v2.targets.rag import RAGEvaluationTarget
from evaluation.v2.targets.tool import ToolEvaluationTarget
from evaluation.v2.targets.workflow import WorkflowEvaluationTarget
from evaluation.v2.threshold import ThresholdEvaluator
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class EvalToolArgs(BaseModel):
    text: str | None = None


class EvalEchoTool(BaseTool):
    name = "evaluation_v2_echo_test"
    description = "evaluation v2 echo"
    args_schema = EvalToolArgs
    source = "test"

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"text": arguments.get("text")})

    async def arun(self, arguments: dict) -> ToolResult:
        return self.run(arguments)


class FakeTarget:
    name = "fake"
    active = 0
    max_active = 0

    async def arun(
        self, case: EvaluationCase, context: EvaluationContext
    ) -> EvaluationTargetResult:
        FakeTarget.active += 1
        FakeTarget.max_active = max(FakeTarget.max_active, FakeTarget.active)
        await asyncio.sleep(float(case.input.get("delay", 0)))
        FakeTarget.active -= 1
        if case.input.get("raise"):
            raise RuntimeError("target error")
        return EvaluationTargetResult(
            target=case.target,
            answer=case.input.get("answer", "ok"),
            sources=case.input.get("sources", []),
            chunks=case.input.get("chunks", []),
            tool_calls=case.input.get("tool_calls", []),
            observations=case.input.get("observations", []),
            trace=case.input.get("trace", []),
            output=case.input.get("output", {}),
            metadata=case.input.get("metadata", {}),
            usage=case.input.get("usage", {}),
            duration_ms=10,
        )


def make_case(
    case_id: str = "case_1",
    target: str = "rag",
    metrics: list[str] | None = None,
    thresholds: dict[str, Any] | None = None,
    **kwargs: Any,
) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        name=case_id,
        target=target,
        query=kwargs.pop("query", "hello"),
        input=kwargs.pop("input", {}),
        expected=kwargs.pop("expected", {}),
        metrics=metrics or [],
        thresholds=thresholds or {},
        **kwargs,
    )


def make_suite(cases: list[EvaluationCase]) -> EvaluationSuite:
    return EvaluationSuite(id="suite", name="suite", cases=cases, concurrency=2, timeout_seconds=1)


def fake_report(case_metric_value: float = 1.0):
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return build_report(
        run_id="run-test",
        suite_id="suite",
        suite_name="suite",
        suite_version="1.0",
        started_at=now,
        finished_at=now,
        cases=[
            EvaluationCaseResult(
                case_id="case_1",
                target="rag",
                status="passed",
                passed=True,
                duration_ms=1,
                metrics=[
                    MetricResult(
                        name="keyword_coverage",
                        value=case_metric_value,
                        passed=True,
                    )
                ],
            )
        ],
    )


def test_old_evaluation_case_still_loads():
    suite = load_v1_questions_as_suite(Path("evaluation/datasets/questions.yaml"))
    assert suite.id == "evaluation_v1_compat"
    assert suite.cases


def test_evaluation_case_v2_validation():
    case = make_case(target="agent")
    assert case.target == "agent"


def test_suite_loads_yaml():
    suite = load_suite("evaluation/v2/fixtures/suites/smoke.yaml")
    assert suite.id == "smoke"
    assert len(suite.cases) >= 5


def test_suite_rejects_invalid_target():
    with pytest.raises(ValueError):
        make_case(target="bad")


def test_rag_target_uses_existing_pipeline(monkeypatch):
    called = {"pipeline": False}

    class FakePipeline:
        def run(self, context):
            called["pipeline"] = True
            context.context_chunks = []
            context.metadata = {"fake": True}
            return context

    monkeypatch.setattr("evaluation.v2.targets.rag.RetrieverPipeline", FakePipeline)
    result = asyncio.run(
        RAGEvaluationTarget().arun(
            make_case(input={"mode": "retrieval-only"}),
            EvaluationContext(run_id="r", suite_id="s"),
        )
    )
    assert called["pipeline"] is True
    assert result.metadata["fake"] is True


def test_agent_target_uses_agent_runtime_factory(monkeypatch):
    class FakeRuntime:
        async def arun(self, request):
            return type(
                "Result",
                (),
                {
                    "answer": "agent ok",
                    "action": "direct_answer",
                    "sources": [],
                    "citations": [],
                    "tool_calls": [],
                    "observations": [],
                    "trace": [],
                    "metadata": {},
                },
            )()

    monkeypatch.setattr(
        "evaluation.v2.targets.agent.AgentRuntimeFactory.get_runtime", lambda: FakeRuntime()
    )
    result = asyncio.run(
        AgentEvaluationTarget().arun(
            make_case(target="agent"), EvaluationContext(run_id="r", suite_id="s")
        )
    )
    assert result.answer == "agent ok"


def test_tool_target_uses_tool_executor():
    registry = get_tool_registry()
    if registry.contains(EvalEchoTool.name):
        registry.unregister(EvalEchoTool.name)
    registry.register(EvalEchoTool())
    result = asyncio.run(
        ToolEvaluationTarget().arun(
            make_case(
                target="tool", input={"tool_name": EvalEchoTool.name, "arguments": {"text": "ok"}}
            ),
            EvaluationContext(run_id="r", suite_id="s"),
        )
    )
    registry.unregister(EvalEchoTool.name)
    assert result.output["success"] is True
    assert "arguments_hash" in result.input


def test_mcp_target_uses_tool_executor(monkeypatch):
    descriptor = ToolDescriptor(name="mcp__x__echo", description="x", provider="mcp", enabled=True)
    monkeypatch.setattr(
        "evaluation.v2.targets.mcp.get_tool_registry",
        lambda: type("R", (), {"get_descriptor": lambda self, name: descriptor})(),
    )
    called = {"tool": False}

    async def fake_arun(self, case, context):
        called["tool"] = True
        return EvaluationTargetResult(target="mcp", metadata={"mcp_tool_available": True})

    monkeypatch.setattr("evaluation.v2.targets.tool.ToolEvaluationTarget.arun", fake_arun)
    from evaluation.v2.targets.mcp import MCPEvaluationTarget

    result = asyncio.run(
        MCPEvaluationTarget().arun(
            make_case(target="mcp", input={"tool_name": "mcp__x__echo"}),
            EvaluationContext(run_id="r", suite_id="s"),
        )
    )
    assert called["tool"] is True
    assert result.metadata["mcp_tool_available"] is True


def test_workflow_target_uses_workflow_runtime_factory(monkeypatch):
    class FakeRuntime:
        async def arun(self, request):
            return type(
                "Result",
                (),
                {"answer": "wf", "output": {"status": "completed"}, "trace": [], "metadata": {}},
            )()

    monkeypatch.setattr(
        "evaluation.v2.targets.workflow.WorkflowRuntimeFactory.get_runtime", lambda: FakeRuntime()
    )
    result = asyncio.run(
        WorkflowEvaluationTarget().arun(
            make_case(target="workflow"), EvaluationContext(run_id="r", suite_id="s")
        )
    )
    assert result.answer == "wf"


def test_retriever_hit_metric():
    case = make_case(expected={"documents": ["劳动法.pdf"]})
    result = EvaluationTargetResult(target="rag", sources=[{"source": "中国劳动法.pdf"}])
    value, _ = RetrieverHitMetric().compute(case, result)
    assert value is True


def test_chunk_recall_metric():
    case = make_case(expected={"chunk_indexes": [1, 2, 3]})
    result = EvaluationTargetResult(target="rag", chunks=[{"chunk_index": 1}, {"chunk_index": 3}])
    value, _ = ChunkRecallMetric().compute(case, result)
    assert value == pytest.approx(2 / 3)


def test_keyword_coverage_metric():
    case = make_case(expected={"keywords": ["a", "b"]})
    result = EvaluationTargetResult(target="generation", answer="a")
    value, _ = KeywordCoverageMetric().compute(case, result)
    assert value == 0.5


def test_mrr_metric():
    metric = get_metric("mrr")
    value, _ = metric.compute(
        make_case(expected={"chunk_indexes": [3]}),
        EvaluationTargetResult(target="rag", chunks=[{"chunk_index": 1}, {"chunk_index": 3}]),
    )
    assert value == 0.5


def test_hit_rate_at_k_metric():
    metric = get_metric("hit_rate_at_k")
    value, _ = metric.compute(
        make_case(expected={"chunk_indexes": [2]}, thresholds={"hit_rate_at_k": {"k": 1}}),
        EvaluationTargetResult(target="rag", chunks=[{"chunk_index": 2}]),
    )
    assert value == 1.0


def test_context_precision_proxy():
    metric = get_metric("context_precision_proxy")
    value, details = metric.compute(
        make_case(expected={"keywords": ["劳动"]}),
        EvaluationTargetResult(target="rag", chunks=[{"text": "劳动"}, {"text": "安卓"}]),
    )
    assert value == 0.5
    assert details["proxy"] is True


def test_exact_match_metric():
    result = EvaluationTargetResult(target="generation", answer="ok")
    metric_result = asyncio.run(
        get_metric("exact_match").evaluate(
            make_case(expected={"answer": "ok"}),
            result,
            EvaluationContext(run_id="r", suite_id="s"),
        )
    )
    assert metric_result.passed is True


def test_contains_keywords_metric():
    value, _ = get_metric("contains_keywords").compute(
        make_case(expected={"keywords": ["ok"]}),
        EvaluationTargetResult(target="generation", answer="ok"),
    )
    assert value is True


def test_tool_selection_accuracy():
    value, _ = get_metric("tool_selection_accuracy").compute(
        make_case(expected={"tools": ["calculator"]}),
        EvaluationTargetResult(target="agent", tool_calls=[{"tool_name": "calculator"}]),
    )
    assert value == 1.0


def test_tool_call_success_rate():
    value, _ = get_metric("tool_call_success_rate").compute(
        make_case(expected={"tools": ["x"]}),
        EvaluationTargetResult(
            target="agent", observations=[{"success": True}, {"success": False}]
        ),
    )
    assert value == 0.5


def test_unnecessary_tool_calls():
    value, _ = get_metric("unnecessary_tool_calls").compute(
        make_case(expected={"tools": ["calculator"]}),
        EvaluationTargetResult(target="agent", tool_calls=[{"tool_name": "echo"}]),
    )
    assert value == 1


def test_mcp_tool_available():
    value, _ = get_metric("mcp_tool_available").compute(
        make_case(target="mcp"),
        EvaluationTargetResult(target="mcp", metadata={"mcp_tool_available": True}),
    )
    assert value is True


def test_workflow_status_match():
    value, _ = get_metric("workflow_status_match").compute(
        make_case(target="workflow", expected={"status": "completed"}),
        EvaluationTargetResult(target="workflow", output={"status": "completed"}),
    )
    assert value is True


def test_node_path_match():
    value, _ = get_metric("node_path_match").compute(
        make_case(target="workflow", expected={"node_path": ["a", "b"]}),
        EvaluationTargetResult(target="workflow", trace=[{"node_id": "a"}, {"node_id": "b"}]),
    )
    assert value is True


def test_threshold_min():
    assert ThresholdEvaluator().evaluate(0.7, {"min": 0.6}) is True


def test_threshold_max():
    assert ThresholdEvaluator().evaluate(10, {"max": 5}) is False


def test_threshold_boolean():
    assert ThresholdEvaluator().evaluate(True, True) is True


def test_case_fails_when_required_metric_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    case = make_case(
        metrics=["keyword_coverage"],
        expected={"keywords": ["missing"]},
        thresholds={"keyword_coverage": {"min": 1}},
        input={"answer": "ok"},
    )
    result = asyncio.run(EvaluationRunnerV2().arun_case(case, make_suite([case])))
    assert result.status == "failed"


def test_suite_pass_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    report = EvaluationRunnerV2().run_suite(make_suite([make_case(input={"answer": "ok"})]))
    assert report.pass_rate == 1.0


def test_baseline_save_and_load(tmp_path):
    report = fake_report()
    manager = BaselineManager(tmp_path)
    manager.save_baseline("base", report)
    assert manager.load_baseline("base").run_id == report.run_id


def test_baseline_name_path_traversal_rejected(tmp_path):
    with pytest.raises(EvaluationBaselineError):
        BaselineManager(tmp_path).save_baseline("../bad", fake_report())


def test_regression_higher_is_better():
    base = fake_report().model_dump(mode="json")
    current = fake_report().model_dump(mode="json")
    base["cases"][0]["metrics"][0]["value"] = 0.2
    current["cases"][0]["metrics"][0]["value"] = 0.8
    summary = RegressionComparator().compare(current, base)
    assert summary["improved_metrics"] >= 1


def test_regression_lower_is_better():
    base = fake_report().model_dump(mode="json")
    current = fake_report().model_dump(mode="json")
    base["cases"][0]["metrics"][0]["name"] = "latency_ms"
    current["cases"][0]["metrics"][0]["name"] = "latency_ms"
    base["cases"][0]["metrics"][0]["value"] = 100
    current["cases"][0]["metrics"][0]["value"] = 50
    summary = RegressionComparator().compare(current, base)
    assert summary["improved_metrics"] == 1


def test_regression_tolerance():
    base = fake_report().model_dump(mode="json")
    current = fake_report().model_dump(mode="json")
    summary = RegressionComparator().compare(current, base)
    assert summary["unchanged_metrics"] >= 1


def test_history_report_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    report = fake_report()
    assert HistoryManager(tmp_path).save(report).exists()


def test_latest_report_compat_written(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    fake_report()
    assert Path("evaluation/report.json").exists()


def test_report_redacts_sensitive_fields():
    redacted = redact_and_truncate({"Authorization": "secret"})
    assert redacted["Authorization"] == "***REDACTED***"


def test_report_truncates_large_text():
    assert redact_and_truncate("abcdef", max_chars=3) == "abc...[truncated]"


def test_cost_unknown_model_is_null():
    value, details = get_metric("estimated_cost").compute(
        make_case(),
        EvaluationTargetResult(
            target="generation",
            metadata={"provider": "x", "model": "y"},
            usage={"total_tokens": 10},
        ),
    )
    assert value is None
    assert details["available"] is False


def test_cost_known_model():
    from evaluation.v2.cost import CostCalculator

    estimate = CostCalculator().estimate(
        "dummy", "dummy-llm", {"input_tokens": 1000, "output_tokens": 1000}
    )
    assert estimate["estimated_cost"] is not None


def test_usage_unavailable_is_not_faked():
    value, details = get_metric("token_usage").compute(
        make_case(), EvaluationTargetResult(target="generation")
    )
    assert value is None
    assert details["available"] is False


def test_llm_judge_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_LLM_JUDGE_ENABLED", False)
    from evaluation.v2.judges.llm import LLMJudge

    result = asyncio.run(LLMJudge().judge(make_case(), EvaluationTargetResult(target="generation")))
    assert result["enabled"] is False


def test_runner_executes_cases_concurrently(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    FakeTarget.max_active = 0
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    cases = [make_case(f"c{i}", input={"delay": 0.02}) for i in range(4)]
    EvaluationRunnerV2(concurrency=4).run_suite(
        EvaluationSuite(id="s", name="s", cases=cases, concurrency=4)
    )
    assert FakeTarget.max_active > 1


def test_runner_respects_concurrency_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    FakeTarget.max_active = 0
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    cases = [make_case(f"c{i}", input={"delay": 0.01}) for i in range(4)]
    EvaluationRunnerV2(concurrency=1).run_suite(
        EvaluationSuite(id="s", name="s", cases=cases, concurrency=1)
    )
    assert FakeTarget.max_active == 1


def test_case_timeout(monkeypatch):
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    case = make_case(timeout_seconds=0.001, input={"delay": 0.05})
    result = asyncio.run(EvaluationRunnerV2().arun_case(case, make_suite([case])))
    assert result.status == "failed"
    assert "timeout" in str(result.error)


def test_case_error_does_not_stop_suite(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    cases = [make_case("bad", input={"raise": True}), make_case("ok")]
    report = EvaluationRunnerV2().run_suite(make_suite(cases))
    assert report.total_cases == 2


def test_runner_cancellation_propagates(monkeypatch):
    class CancelTarget:
        async def arun(self, case, context):
            raise asyncio.CancelledError()

    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: CancelTarget())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(EvaluationRunnerV2().arun_case(make_case(), make_suite([make_case()])))


def test_evaluation_run_api(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr("evaluation.v2.runner.get_target", lambda target: FakeTarget())
    app = FastAPI()
    app.include_router(evaluation_router)
    response = TestClient(app).post("/evaluation/run", json={"suite": "smoke", "tags": ["tool"]})
    assert response.status_code == 200
    assert response.json()["code"] == 0


def test_evaluation_reports_api(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    report = fake_report()
    HistoryManager(tmp_path).save(report)
    app = FastAPI()
    app.include_router(evaluation_router)
    client = TestClient(app)
    assert client.get("/evaluation/reports?suite_id=suite").status_code == 200
    assert client.get(f"/evaluation/reports/{report.run_id}?suite_id=suite").status_code == 200


def test_debug_evaluation_api():
    from backend.app.api.evaluation import evaluation_debug_state

    state = evaluation_debug_state()
    assert "targets" in state
    assert "metrics" in state


def test_v1_evaluation_still_runs(monkeypatch, tmp_path):
    class FakeChatService:
        def chat(self, request):
            return type(
                "Response",
                (),
                {
                    "answer": "劳动合同 试用期 劳动合同解除",
                    "sources": [{"source": "中国劳动法.pdf", "chunk_index": 2}],
                    "metadata": {},
                },
            )()

    from evaluation.run import run_evaluation

    report = run_evaluation(report_path=tmp_path / "report.json", chat_service=FakeChatService())
    assert "total" in report


def test_chat_agent_workflow_apis_still_exist():
    from backend.app.api import agent_router, chat_router, workflow_router

    paths = {
        route.path
        for router in (chat_router, agent_router, workflow_router)
        for route in router.routes
        if hasattr(route, "path")
    }
    assert "/chat" in paths
    assert "/agent/chat" in paths
    assert "/workflow/run" in paths


def test_real_evaluation_v2_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_DIR", str(tmp_path))
    registry = get_tool_registry()
    if registry.contains(EvalEchoTool.name):
        registry.unregister(EvalEchoTool.name)
    registry.register(EvalEchoTool())
    suite = EvaluationSuite(
        id="real_smoke",
        name="real smoke",
        cases=[
            make_case(
                "tool_real",
                target="tool",
                input={"tool_name": EvalEchoTool.name, "arguments": {"text": "hello"}},
                metrics=["tool_success", "result_contains"],
                expected={"contains": "hello"},
                thresholds={"tool_success": True},
            )
        ],
    )
    report = EvaluationRunnerV2().run_suite(suite)
    registry.unregister(EvalEchoTool.name)
    assert report.passed is True
    assert (tmp_path / "report.json").exists()


def test_metric_registry_contains_required_metrics():
    names = list_metrics()
    assert "retriever_hit" in names
    assert "workflow_status_match" in names


def test_empty_answer_metric():
    value, _ = get_metric("empty_answer").compute(
        make_case(), EvaluationTargetResult(target="generation", answer="")
    )
    assert value is True


def test_source_coverage_metric():
    value, _ = get_metric("source_coverage").compute(
        make_case(expected={"sources": ["a.pdf"]}),
        EvaluationTargetResult(target="generation", sources=[{"source": "a.pdf"}]),
    )
    assert value == 1.0
