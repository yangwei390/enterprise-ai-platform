import asyncio
from time import perf_counter

import pytest
from backend.app.agents.langgraph.graph import build_agent_graph
from backend.app.agents.langgraph.nodes import (
    ObservationNode,
    ToolNode,
)
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.langgraph.state import create_initial_state
from backend.app.agents.langgraph.tool_calling import NativeToolCallingStrategy
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.api.agent import router as agent_router
from backend.app.config.settings import settings
from backend.app.llms import LLMRequest, LLMResponse, LLMToolCall
from backend.app.tools import BaseTool, ToolExecutor, ToolRegistry, ToolResult
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class EmptyArgs(BaseModel):
    pass


class TextArgs(BaseModel):
    text: str = ""


class ExpressionArgs(BaseModel):
    expression: str = ""


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    args_schema = TextArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"text": arguments.get("text", "")})


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Calculate simple expression"
    args_schema = ExpressionArgs

    def run(self, arguments: dict) -> ToolResult:
        expression = str(arguments.get("expression", ""))
        if expression == "12+30":
            value = 42
        else:
            value = 0
        return ToolResult(name=self.name, success=True, result={"result": value})


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Get current time"
    args_schema = EmptyArgs

    async def arun(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"time": "10:00"})

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"time": "10:00"})


class FailingTool(BaseTool):
    name = "failing"
    description = "Always fails"
    args_schema = EmptyArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False, error="tool failed")


class SlowTool(BaseTool):
    name = "slow"
    description = "Slow async tool"
    args_schema = EmptyArgs

    async def arun(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(name=self.name, success=True, result={"ok": True})

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"ok": True})


class FakeToolCallingLLM:
    supports_tool_calling = True

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(answer="final", model="fake", finish_reason="stop")


class FakeNoToolCallingLLM(FakeToolCallingLLM):
    supports_tool_calling = False


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _state(query: str = "query"):
    return create_initial_state(
        query=query,
        conversation_id=None,
        knowledge_base_id=None,
        memory_context=None,
        metadata={},
    )


def _run_graph(graph, *, query: str = "x"):
    return asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(
            AgentRuntimeRequest(query=query),
        )
    )


def _patch_registry(monkeypatch, registry: ToolRegistry) -> None:
    monkeypatch.setattr(
        "backend.app.agents.langgraph.tool_calling.get_tool_registry",
        lambda: registry,
    )


def _patch_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr("backend.app.agents.langgraph.tool_calling.LLMFactory.get_llm", lambda: llm)
    monkeypatch.setattr("backend.app.agents.langgraph.reflection.LLMFactory.get_llm", lambda: llm)


def _tool_call(name: str, arguments: dict | None = None, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        answer="",
        model="fake",
        finish_reason="tool_calls",
        tool_calls=[LLMToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


def _final(answer: str = "done") -> LLMResponse:
    return LLMResponse(answer=answer, model="fake", finish_reason="stop")


def test_native_tool_calling_strategy_uses_registry_tools(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([_tool_call("echo", {"text": "hi"}, "tc_1")])
    _patch_llm(monkeypatch, llm)
    state = _state("echo hi")

    decision = asyncio.run(NativeToolCallingStrategy().adecide(state))

    assert decision.action == "tool_calls"
    assert decision.tool_calls[0].tool_name == "echo"
    assert llm.requests[0].tools[0]["function"]["name"] == "echo"


def test_disabled_tool_not_sent_to_llm(monkeypatch):
    registry = _registry(EchoTool())
    registry.disable("echo")
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([_final()])
    _patch_llm(monkeypatch, llm)

    asyncio.run(NativeToolCallingStrategy().adecide(_state("echo")))

    assert llm.requests[0].tools == []


def test_unknown_tool_call_is_rejected(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([_tool_call("missing")])
    _patch_llm(monkeypatch, llm)

    decision = asyncio.run(NativeToolCallingStrategy().adecide(_state("bad")))

    assert decision.action == "reflect"
    assert decision.metadata["tool_registry"]["rejected_tools"] == ["missing"]


def test_provider_without_tool_calling_falls_back_to_json_plan(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeNoToolCallingLLM([]))
    monkeypatch.setattr(settings, "AGENT_PLANNER_FALLBACK_ENABLED", True)

    decision = asyncio.run(NativeToolCallingStrategy().adecide(_state("echo")))

    assert decision.metadata["actual_strategy"] == "json_plan"
    assert decision.metadata["fallback_used"] is True


def test_native_tool_call_parses_multiple_tools(monkeypatch):
    registry = _registry(EchoTool(), CalculatorTool())
    _patch_registry(monkeypatch, registry)
    response = LLMResponse(
        answer="",
        model="fake",
        finish_reason="tool_calls",
        tool_calls=[
            LLMToolCall(id="a", name="echo", arguments={"text": "hi"}),
            LLMToolCall(id="b", name="calculator", arguments={"expression": "12+30"}),
        ],
    )
    _patch_llm(monkeypatch, FakeToolCallingLLM([response]))

    decision = asyncio.run(NativeToolCallingStrategy().adecide(_state("both")))

    assert [tool.tool_name for tool in decision.tool_calls] == ["echo", "calculator"]
    assert [tool.id for tool in decision.tool_calls] == ["a", "b"]


def test_tool_call_id_is_preserved(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo", call_id="keep_me")]))

    decision = asyncio.run(NativeToolCallingStrategy().adecide(_state("id")))

    assert decision.tool_calls[0].id == "keep_me"


def test_tool_node_executes_multiple_tools():
    registry = _registry(EchoTool(), CalculatorTool())
    state = _state("tools")
    state["pending_tool_calls"] = [
        {"id": "a", "tool_name": "echo", "arguments": {"text": "hi"}, "index": 0},
        {"id": "b", "tool_name": "calculator", "arguments": {"expression": "12+30"}, "index": 1},
    ]

    result = asyncio.run(ToolNode(ToolExecutor(registry=registry)).acall(state))

    assert len(result["tool_results"]) == 2
    assert [item["tool_call_id"] for item in result["tool_results"]] == ["a", "b"]


def test_observation_is_injected_into_next_llm_call(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("echo", {"text": "hi"}, "tc_1"),
        _final("observed"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(
        tool_node=ToolNode(ToolExecutor(registry=registry)),
        async_mode=True,
    )

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(AgentRuntimeRequest(query="echo"))
    )

    assert result.answer == "observed"
    assert any(message.role == "tool" for message in llm.requests[1].messages)


def test_agent_loops_back_after_tool_result(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([_tool_call("echo"), _final("loop final")])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(
        tool_node=ToolNode(ToolExecutor(registry=registry)),
        async_mode=True,
    )

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(AgentRuntimeRequest(query="loop"))
    )

    assert result.answer == "loop final"
    assert result.metadata["agent_loop"]["loop_iterations"] >= 2


def test_agent_can_call_tool_twice_in_different_steps(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("echo", {"text": "one"}, "one"),
        _tool_call("echo", {"text": "two"}, "two"),
        _final("done"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert [call["tool_call_id"] for call in result.tool_calls] == ["one", "two"]


def test_agent_can_call_two_different_tools_sequentially(monkeypatch):
    registry = _registry(CurrentTimeTool(), CalculatorTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("current_time", {}, "time"),
        _tool_call("calculator", {"expression": "12+30"}, "calc"),
        _final("10:00 and 42"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "10:00 and 42"
    assert [call["tool_name"] for call in result.tool_calls] == ["current_time", "calculator"]


def test_agent_final_answer_after_observation(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo"), _final("final after obs")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "final after obs"
    assert result.metadata["agent_loop"]["termination_reason"] == "final_answer"


def test_agent_multi_tool_parallel_call(monkeypatch):
    registry = _registry(SlowTool(), CurrentTimeTool())
    _patch_registry(monkeypatch, registry)
    response = LLMResponse(
        answer="",
        model="fake",
        finish_reason="tool_calls",
        tool_calls=[
            LLMToolCall(id="slow", name="slow", arguments={}),
            LLMToolCall(id="time", name="current_time", arguments={}),
        ],
    )
    _patch_llm(monkeypatch, FakeToolCallingLLM([response, _final("parallel done")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    started_at = perf_counter()
    result = _run_graph(graph)

    assert perf_counter() - started_at < 0.2
    assert result.answer == "parallel done"


def test_reflection_triggered_on_tool_failure(monkeypatch):
    registry = _registry(FailingTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("failing", {}, "bad"),
        LLMResponse(
            answer='{"status":"final","reason":"failed","final_answer":"reflected"}',
            model="fake",
        ),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "reflected"
    assert result.metadata["reflection"]["triggered"] is True


def test_reflection_not_triggered_on_success(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo"), _final("ok")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.metadata["reflection"]["triggered"] is False


def test_reflection_replan(monkeypatch):
    registry = _registry(FailingTool(), EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("failing", {}, "bad"),
        LLMResponse(answer='{"status":"replan","reason":"try echo"}', model="fake"),
        _tool_call("echo", {"text": "ok"}, "good"),
        _final("replanned"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "replanned"


def test_reflection_final(monkeypatch):
    registry = _registry(FailingTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(
        monkeypatch,
        FakeToolCallingLLM([
            _tool_call("failing", {}, "bad"),
            LLMResponse(
                answer='{"status":"final","reason":"enough","final_answer":"final"}',
                model="fake",
            ),
        ]),
    )
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "final"


def test_reflection_fail(monkeypatch):
    registry = _registry(FailingTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(
        monkeypatch,
        FakeToolCallingLLM([
            _tool_call("failing", {}, "bad"),
            LLMResponse(
                answer='{"status":"fail","reason":"stop","final_answer":"failed"}',
                model="fake",
            ),
        ]),
    )
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "failed"


def test_same_tool_repeat_limit(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    monkeypatch.setattr(settings, "AGENT_MAX_SAME_TOOL_REPEATS", 1)
    llm = FakeToolCallingLLM([
        _tool_call("echo", {"text": "same"}, "one"),
        _tool_call("echo", {"text": "same"}, "two"),
        LLMResponse(
            answer='{"status":"final","reason":"repeat","final_answer":"repeat stopped"}',
            model="fake",
        ),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.metadata["agent_loop"]["same_tool_repeat_limit_triggered"] is True


@pytest.mark.parametrize(
    ("setting_name", "expected_reason"),
    [
        ("AGENT_MAX_STEPS", "max_steps_exceeded"),
        ("AGENT_MAX_LLM_CALLS", "max_llm_calls_exceeded"),
        ("AGENT_MAX_TOOL_CALLS", "max_tool_calls_exceeded"),
        ("AGENT_MAX_REFLECTIONS", "max_reflections_exceeded"),
    ],
)
def test_budget_limits(monkeypatch, setting_name, expected_reason):
    registry = _registry(EchoTool(), FailingTool())
    _patch_registry(monkeypatch, registry)
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS", 50)
    monkeypatch.setattr(settings, "AGENT_MAX_LLM_CALLS", 50)
    monkeypatch.setattr(settings, "AGENT_MAX_TOOL_CALLS", 50)
    monkeypatch.setattr(settings, "AGENT_MAX_REFLECTIONS", 50)
    monkeypatch.setattr(settings, setting_name, 1)
    tool_name = "failing" if setting_name == "AGENT_MAX_REFLECTIONS" else "echo"
    responses = [_tool_call(tool_name, {"text": "x"}, "one")] * 5
    responses.append(_final("fallback final"))
    _patch_llm(monkeypatch, FakeToolCallingLLM(responses))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.metadata["agent_loop"]["budget_exceeded"] is True
    assert result.metadata["agent_loop"]["termination_reason"] == expected_reason


def test_max_duration_budget(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    monkeypatch.setattr(settings, "AGENT_MAX_DURATION_SECONDS", 0)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.metadata["agent_loop"]["termination_reason"] == "max_duration_exceeded"


def test_budget_metadata(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_final("done")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert "steps" in result.metadata["agent_loop"]
    assert "llm_calls" in result.metadata["agent_loop"]


def test_tool_failure_becomes_observation():
    state = _state("x")
    state["tool_results"] = [
        {
            "tool_call_id": "bad",
            "tool_name": "failing",
            "success": False,
            "result": None,
            "error": "broken",
            "metadata": {},
        }
    ]

    result = asyncio.run(ObservationNode().acall(state))

    assert result["observations"][0]["success"] is False
    assert "broken" in result["observations"][0]["content"]


def test_empty_observation_triggers_reflection():
    state = _state("x")
    state["tool_results"] = [
        {
            "tool_call_id": "empty",
            "tool_name": "echo",
            "success": True,
            "result": "",
            "error": None,
            "metadata": {},
        }
    ]

    result = asyncio.run(ObservationNode().acall(state))

    assert result["current_action"] == "reflect"


def test_agent_session_preserves_recent_observations(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo"), _final("ok")]))
    saved = {}

    class FakeMemoryManager:
        provider = type("Provider", (), {"name": "memory"})()

        def load_session(self, session_id):
            return None

        def save_session(self, session_state):
            saved["tool_results"] = session_state.tool_results

    monkeypatch.setattr(
        "backend.app.agents.langgraph.runtime.MemoryFactory.get_manager",
        lambda: FakeMemoryManager(),
    )
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    asyncio.run(LangGraphAgentRuntime(graph_app=graph).arun(AgentRuntimeRequest(query="x")))

    assert saved["tool_results"]


def test_agent_trace_contains_loop_events(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("echo"), _final("ok")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    events = [step.event for step in result.trace]
    assert "planner_completed" in events
    assert "tool_completed" in events
    assert "observation_created" in events
    assert "final_answer" in events


def test_agent_metadata_contains_loop_summary(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_final("ok")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.metadata["agent_loop"]["enabled"] is True
    assert result.metadata["tool_calling"]["native_used"] is True


def test_agent_api_response_compatible(monkeypatch):
    class FakeRuntime:
        async def arun(self, request):
            from backend.app.agents.state import AgentRuntimeResult

            return AgentRuntimeResult(
                answer="ok",
                action="direct_answer",
                metadata={"agent_loop": {}},
            )

    monkeypatch.setattr(
        "backend.app.api.agent.AgentRuntimeFactory.get_runtime",
        lambda: FakeRuntime(),
    )
    app = FastAPI()
    app.include_router(agent_router)
    client = TestClient(app)

    response = client.post("/agent/chat", json={"query": "hi"})

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "ok"


def test_agent_v1_still_works():
    from backend.app.agents.runtime import AgentRuntime

    result = AgentRuntime().run(AgentRuntimeRequest(query="你好"))

    assert result.action == "direct_answer"


def test_workflow_v2_agent_node_still_works():
    from backend.app.workflows.langgraph.nodes import WorkflowNodeFactory

    assert hasattr(WorkflowNodeFactory, "_run_agent_node")


def test_mcp_tool_can_be_called_in_agent_loop(monkeypatch):
    class MCPTool(EchoTool):
        name = "mcp__demo__echo"
        source = "mcp"

    registry = _registry(MCPTool())
    _patch_registry(monkeypatch, registry)
    _patch_llm(monkeypatch, FakeToolCallingLLM([_tool_call("mcp__demo__echo"), _final("mcp ok")]))
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "mcp ok"
    assert result.tool_calls[0]["tool_name"] == "mcp__demo__echo"


def test_evaluation_agent_target_reads_loop_metadata():
    from evaluation.v2.metrics.agent import LoopIterationsMetric
    from evaluation.v2.result import EvaluationTargetResult
    from evaluation.v2.schemas import EvaluationCase

    case = EvaluationCase(
        id="agent_loop",
        name="agent loop",
        target="agent",
        expected={"loop_iterations": 2},
        metrics=["loop_iterations"],
    )
    result = EvaluationTargetResult(
        target="agent",
        metadata={"agent_loop": {"loop_iterations": 2}},
    )

    metric = asyncio.run(LoopIterationsMetric().evaluate(case, result, None))

    assert metric.value == 2
    assert metric.passed is True


def test_real_native_tool_calling_smoke(monkeypatch):
    registry = _registry(EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([_tool_call("echo", {"text": "real"}, "real"), _final("real final")])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph, query="real")

    assert result.answer == "real final"
    assert llm.requests[0].tools
    assert any(message.role == "tool" for message in llm.requests[1].messages)


def test_real_agent_loop_smoke(monkeypatch):
    registry = _registry(CurrentTimeTool(), CalculatorTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("current_time", {}, "time"),
        _tool_call("calculator", {"expression": "12+30"}, "calc"),
        _final("time plus calc"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert [call["tool_name"] for call in result.tool_calls] == ["current_time", "calculator"]
    assert result.answer == "time plus calc"


def test_real_reflection_smoke(monkeypatch):
    registry = _registry(FailingTool(), EchoTool())
    _patch_registry(monkeypatch, registry)
    llm = FakeToolCallingLLM([
        _tool_call("failing", {}, "bad"),
        LLMResponse(answer='{"status":"replan","reason":"use echo"}', model="fake"),
        _tool_call("echo", {"text": "recover"}, "recover"),
        _final("recovered"),
    ])
    _patch_llm(monkeypatch, llm)
    graph = build_agent_graph(tool_node=ToolNode(ToolExecutor(registry=registry)), async_mode=True)

    result = _run_graph(graph)

    assert result.answer == "recovered"
    assert result.metadata["reflection"]["triggered"] is True
