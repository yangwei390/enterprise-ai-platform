import asyncio

from backend.app.agents.langgraph.nodes import ToolNode
from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.langgraph.state import create_initial_state
from backend.app.agents.langgraph.tool_calling import NativeToolCallingStrategy
from backend.app.llms import LLMRequest, LLMResponse
from backend.app.tools import BaseTool, ToolExecutor, ToolRegistry, ToolResult
from backend.app.tools.providers.workflow import WorkflowTool
from pydantic import BaseModel


class EmptyArgs(BaseModel):
    pass


class QueryArgs(BaseModel):
    query: str


class RequiredValueArgs(BaseModel):
    value: int


class RecordingTool(BaseTool):
    name = "echo"
    description = "Echo text"
    args_schema = QueryArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(name=self.name, success=True, result={"query": arguments["query"]})


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Search knowledge"
    args_schema = QueryArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"answer": arguments["query"]})


class RequiredValueTool(BaseTool):
    name = "requires_value"
    description = "Requires value"
    args_schema = RequiredValueArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(name=self.name, success=True, result={"value": arguments["value"]})


class FinalLLM:
    supports_tool_calling = True

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(answer="done", model="fake")


class JsonPlanLLM:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def chat(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            answer='{"steps":[{"tool":"echo","args":{"query":"x"}},'
            '{"tool":"knowledge_search","args":{"query":"x"}}]}',
            model="fake",
        )


class FakeWorkflowRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return type(
            "WorkflowResult",
            (),
            {
                "status": "completed",
                "metadata": {},
                "model_dump": lambda self: {"answer": "workflow"},
            },
        )()


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _state(*, allowed_tools: list[str], allowed_workflows: list[str] | None = None):
    return create_initial_state(
        query="query",
        conversation_id=None,
        knowledge_base_id=None,
        memory_context=None,
        metadata={
            "agent_id": "test_agent",
            "agent_definition_version": "test",
            "tool_allowlist": allowed_tools,
            "workflow_allowlist": allowed_workflows or [],
        },
    )


def test_native_planner_only_exposes_tool_scope_tools(monkeypatch) -> None:
    registry = _registry(RecordingTool(), KnowledgeSearchTool())
    llm = FinalLLM()
    monkeypatch.setattr(
        "backend.app.agents.langgraph.tool_calling.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.tool_calling.LLMFactory.get_llm",
        lambda: llm,
    )

    decision = asyncio.run(
        NativeToolCallingStrategy().adecide(
            _state(allowed_tools=["knowledge_search"])
        )
    )

    visible_names = [
        item["function"]["name"]
        for item in llm.requests[0].tools
    ]
    assert visible_names == ["knowledge_search"]
    assert decision.metadata["tool_scope"]["allowed_tools"] == ["knowledge_search"]
    assert decision.metadata["tool_scope"]["visible_tools"] == ["knowledge_search"]


def test_json_planner_normalizes_plan_to_tool_scope(monkeypatch) -> None:
    registry = _registry(RecordingTool(), KnowledgeSearchTool())
    llm = JsonPlanLLM()
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: llm,
    )

    plan = LLMPlanner().plan(query="query", tool_allowlist=["knowledge_search"])

    assert [step.tool for step in plan.steps] == ["knowledge_search"]
    available = plan.metadata["tool_registry"]["available_tools"]
    assert available == ["knowledge_search"]
    payload = llm.requests[0].messages[1].content
    assert "knowledge_search" in payload
    assert "echo" not in payload


def test_tool_node_blocks_unallowed_tool_without_execution() -> None:
    echo = RecordingTool()
    registry = _registry(echo, KnowledgeSearchTool())
    state = _state(allowed_tools=["knowledge_search"])
    state["pending_tool_calls"] = [
        {"id": "call_1", "tool_name": "echo", "arguments": {"query": "x"}, "index": 0}
    ]

    result = asyncio.run(ToolNode(ToolExecutor(registry=registry)).acall(state))

    assert echo.calls == 0
    blocked = result["tool_results"][0]
    assert blocked["status"] == "blocked"
    assert blocked["error"] == "tool_not_allowed"
    assert blocked["metadata"]["error_type"] == "tool_permission_error"
    assert result["metadata"]["tool_scope"]["blocked_tools"] == ["echo"]


def test_tool_node_blocks_unallowed_workflow_without_execution() -> None:
    runtime = FakeWorkflowRuntime()
    registry = _registry(WorkflowTool(runtime=runtime))
    state = _state(
        allowed_tools=["workflow_default_knowledge"],
        allowed_workflows=[],
    )
    state["pending_tool_calls"] = [
        {
            "id": "call_1",
            "tool_name": "workflow_default_knowledge",
            "arguments": {"query": "x"},
            "index": 0,
        }
    ]

    result = asyncio.run(ToolNode(ToolExecutor(registry=registry)).acall(state))

    assert runtime.calls == 0
    blocked = result["tool_results"][0]
    assert blocked["status"] == "blocked"
    assert blocked["error"] == "workflow_not_allowed"
    assert blocked["metadata"]["workflow_id"] == "default_agent_workflow_v2"


def test_tool_node_validates_arguments_before_execution() -> None:
    tool = RequiredValueTool()
    registry = _registry(tool)
    state = _state(allowed_tools=["requires_value"])
    state["pending_tool_calls"] = [
        {"id": "call_1", "tool_name": "requires_value", "arguments": {}, "index": 0}
    ]

    result = asyncio.run(ToolNode(ToolExecutor(registry=registry)).acall(state))

    assert tool.calls == 0
    failed = result["tool_results"][0]
    assert failed["status"] == "validation_failed"
    assert failed["metadata"]["error_type"] == "tool_validation_error"
