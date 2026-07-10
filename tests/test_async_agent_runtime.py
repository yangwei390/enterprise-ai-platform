import asyncio
from time import perf_counter

from backend.app.agents import AgentRuntime, AgentRuntimeResult
from backend.app.agents.langgraph.graph import build_agent_graph
from backend.app.agents.langgraph.nodes import FinalNode, PlannerNode, ToolNode
from backend.app.agents.langgraph.planner import AgentPlan, PlanStep
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.agents.trace import AgentTraceStep
from backend.app.api.agent import router as agent_router
from backend.app.api.chat import get_chat_service
from backend.app.api.chat import router as chat_router
from backend.app.api.workflow import router as workflow_router
from backend.app.chat import ChatRequest, ChatResponse
from backend.app.config.settings import settings
from backend.app.tools import BaseTool, ToolCall, ToolExecutor, ToolResult
from backend.app.tools.registry import ToolRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class AnyArgs(BaseModel):
    query: str | None = None
    text: str | None = None
    expression: str | None = None
    delay: float | None = None
    knowledge_base_id: int | None = None


class AsyncKnowledgeTool(BaseTool):
    name = "knowledge_search"
    description = "async knowledge"
    args_schema = AnyArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False, error="sync path should not run")

    async def arun(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay") or 0))
        return ToolResult(
            name=self.name,
            success=True,
            result={
                "answer": "async knowledge answer",
                "sources": [{"source": "async.pdf"}],
                "citations": [{"source": "async.pdf"}],
                "metadata": {"retriever_mode": "async_smoke"},
            },
        )


class AsyncEchoTool(BaseTool):
    name = "async_echo"
    description = "async echo"
    args_schema = AnyArgs

    def __init__(self) -> None:
        self.called_async = False

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False, error="sync path should not run")

    async def arun(self, arguments: dict) -> ToolResult:
        self.called_async = True
        return ToolResult(
            name=self.name,
            success=True,
            result={"text": arguments.get("text")},
        )


class SyncEchoTool(BaseTool):
    name = "sync_echo"
    description = "sync echo"
    args_schema = AnyArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(
            name=self.name,
            success=True,
            result={"text": arguments.get("text")},
        )


class SlowAsyncTool(BaseTool):
    name = "slow"
    description = "slow async"
    args_schema = AnyArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False)

    async def arun(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay") or 0.2))
        return ToolResult(name=self.name, success=True, result={"done": True})


class FlakyAsyncTool(BaseTool):
    name = "flaky"
    description = "flaky async"
    args_schema = AnyArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False)

    async def arun(self, arguments: dict) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return ToolResult(name=self.name, success=True, result={"attempt": self.calls})


class CountingAsyncTool(BaseTool):
    name = "counting"
    description = "count concurrency"
    args_schema = AnyArgs
    active = 0
    max_active = 0

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False)

    async def arun(self, arguments: dict) -> ToolResult:
        CountingAsyncTool.active += 1
        CountingAsyncTool.max_active = max(
            CountingAsyncTool.max_active,
            CountingAsyncTool.active,
        )
        await asyncio.sleep(0.05)
        CountingAsyncTool.active -= 1
        return ToolResult(name=self.name, success=True, result={"done": True})


class AsyncPlanner:
    def __init__(self, steps: list[PlanStep] | None = None) -> None:
        self.steps = steps or [
            PlanStep(tool="knowledge_search", args={"query": "劳动法"})
        ]

    def plan(self, **kwargs) -> AgentPlan:
        return AgentPlan(steps=self.steps)

    async def aplan(self, **kwargs) -> AgentPlan:
        await asyncio.sleep(0)
        return AgentPlan(steps=self.steps, metadata={"async": True})


class FakeAsyncGraph:
    def __init__(self) -> None:
        self.ainvoke_called = False
        self.invoke_called = False

    def invoke(self, state):
        self.invoke_called = True
        return state

    async def ainvoke(self, state):
        self.ainvoke_called = True
        state["final_answer"] = "ainvoke answer"
        state["metadata"]["trace"].append(
            {
                "step": "final_answer",
                "name": "final_answer",
                "input": {"query": state["query"]},
                "output": {"answer": "ainvoke answer"},
                "duration_ms": 0,
                "async_execution": True,
                "status": "success",
                "error": None,
            }
        )
        return state


class SleepingGraph:
    async def ainvoke(self, state):
        await asyncio.sleep(1)
        return state


class FakeAsyncRuntime:
    called_arun = False
    called_run = False

    async def arun(self, request):
        FakeAsyncRuntime.called_arun = True
        return AgentRuntimeResult(
            answer="async api answer",
            action="direct_answer",
            metadata={"runtime": "fake_async"},
            trace=[
                AgentTraceStep(
                    step="final_answer",
                    name="final_answer",
                    output={"answer": "async api answer"},
                )
            ],
        )

    def run(self, request):
        FakeAsyncRuntime.called_run = True
        return AgentRuntimeResult(answer="sync api answer", action="direct_answer")


class FakeChatService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            query=request.query,
            answer="chat ok",
            sources=[],
            citations=[],
            context_text="",
            prompt_text="",
            metadata={},
        )


class FakeWorkflowRuntime:
    def run(self, request):
        return {
            "answer": "workflow ok",
            "output": {},
            "node_outputs": {},
            "trace": [],
            "metadata": {"failed": False},
        }


class FakeDirectAnswerLLM:
    def chat(self, request):
        return type(
            "Resp",
            (),
            {
                "answer": "ok",
                "model": "fake",
                "metadata": {},
            },
        )()


def test_langgraph_runtime_arun_uses_ainvoke(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ASYNC_ENABLED", True)
    graph = FakeAsyncGraph()

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(
            AgentRuntimeRequest(query="你好")
        )
    )

    assert graph.ainvoke_called is True
    assert graph.invoke_called is False
    assert result.answer == "ainvoke answer"


def test_agent_api_uses_async_runtime(monkeypatch):
    FakeAsyncRuntime.called_arun = False
    FakeAsyncRuntime.called_run = False
    monkeypatch.setattr(
        "backend.app.api.agent.AgentRuntimeFactory.get_runtime",
        lambda: FakeAsyncRuntime(),
    )
    app = FastAPI()
    app.include_router(agent_router)
    client = TestClient(app)

    response = client.post("/agent/chat", json={"query": "你好"})

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "async api answer"
    assert FakeAsyncRuntime.called_arun is True
    assert FakeAsyncRuntime.called_run is False


def test_async_tool_executor_calls_async_tool():
    tool = AsyncEchoTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = asyncio.run(
        ToolExecutor(registry=registry).aexecute(
            ToolCall(name="async_echo", arguments={"text": "hello"})
        )
    )

    assert tool.called_async is True
    assert result.success is True
    assert result.metadata["async_execution"] is True
    assert result.metadata["sync_fallback"] is False


def test_async_tool_executor_falls_back_to_thread(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_SYNC_FALLBACK_ENABLED", True)
    registry = ToolRegistry()
    registry.register(SyncEchoTool())

    result = asyncio.run(
        ToolExecutor(registry=registry).aexecute(
            ToolCall(name="sync_echo", arguments={"text": "hello"})
        )
    )

    assert result.success is True
    assert result.metadata["sync_fallback"] is True


def test_async_tool_timeout(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_TOOL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(settings, "AGENT_TOOL_RETRY_COUNT", 0)
    registry = ToolRegistry()
    registry.register(SlowAsyncTool())

    result = asyncio.run(
        ToolExecutor(registry=registry).aexecute(
            ToolCall(name="slow", arguments={"delay": 0.1})
        )
    )

    assert result.success is False
    assert result.metadata["timeout"] is True
    assert "timed out" in str(result.error)


def test_async_tool_retry(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_TOOL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(settings, "AGENT_TOOL_RETRY_COUNT", 1)
    tool = FlakyAsyncTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = asyncio.run(
        ToolExecutor(registry=registry).aexecute(
            ToolCall(name="flaky", arguments={})
        )
    )

    assert result.success is True
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["retry_count"] == 1


def test_parallel_tool_execution(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_TOOL_MAX_CONCURRENCY", 2)
    registry = ToolRegistry()
    registry.register(SlowAsyncTool())
    tool_node = ToolNode(tool_executor=ToolExecutor(registry=registry))
    steps = [
        {"tool": "slow", "args": {"delay": 0.1}},
        {"tool": "slow", "args": {"delay": 0.1}},
    ]
    state = {
        "messages": [],
        "query": "parallel",
        "plan": {"steps": steps},
        "tool_calls": [],
        "tool_results": [],
        "metadata": {"trace": [], "async_runtime": {}},
    }

    started_at = perf_counter()
    result_state = asyncio.run(tool_node.acall(state))
    duration = perf_counter() - started_at

    assert duration < 0.18
    assert len(result_state["tool_results"]) == 2


def test_tool_concurrency_limit(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_TOOL_MAX_CONCURRENCY", 2)
    CountingAsyncTool.active = 0
    CountingAsyncTool.max_active = 0
    registry = ToolRegistry()
    registry.register(CountingAsyncTool())
    tool_node = ToolNode(tool_executor=ToolExecutor(registry=registry))
    state = {
        "messages": [],
        "query": "limit",
        "plan": {
            "steps": [
                {"tool": "counting", "args": {}},
                {"tool": "counting", "args": {}},
                {"tool": "counting", "args": {}},
                {"tool": "counting", "args": {}},
            ]
        },
        "tool_calls": [],
        "tool_results": [],
        "metadata": {"trace": [], "async_runtime": {}},
    }

    asyncio.run(tool_node.acall(state))

    assert CountingAsyncTool.max_active <= 2


def test_async_runtime_cancellation(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ASYNC_ENABLED", True)

    async def run_and_cancel():
        runtime = LangGraphAgentRuntime(graph_app=SleepingGraph())
        task = asyncio.create_task(runtime.arun(AgentRuntimeRequest(query="cancel")))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(run_and_cancel()) is True


def test_v1_sync_runtime_still_works(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.runtime.LLMFactory.get_llm",
        lambda: FakeDirectAnswerLLM(),
    )

    result = AgentRuntime().run(AgentRuntimeRequest(query="你好"))

    assert result.answer == "ok"
    assert result.trace


def test_chat_and_workflow_api_still_work(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.workflow.WorkflowRuntimeV1",
        lambda: FakeWorkflowRuntime(),
    )
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(workflow_router)
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    chat_response = client.post("/chat", json={"query": "你好"})
    workflow_response = client.post(
        "/workflow/run",
        json={
            "query": "hello",
            "workflow_id": "default_knowledge_workflow",
            "inputs": {},
        },
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["code"] == 0
    assert workflow_response.status_code == 200
    assert workflow_response.json()["code"] == 0


def test_real_langgraph_async_smoke(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ASYNC_ENABLED", True)
    registry = ToolRegistry()
    registry.register(AsyncKnowledgeTool())
    graph_app = build_agent_graph(
        planner_node=PlannerNode(planner=AsyncPlanner()),
        tool_node=ToolNode(tool_executor=ToolExecutor(registry=registry)),
        final_node=FinalNode(),
        async_mode=True,
    )

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph_app).arun(
            AgentRuntimeRequest(query="劳动法第二章说什么", knowledge_base_id=4)
        )
    )

    assert result.answer == "async knowledge answer"
    assert result.sources[0]["source"] == "async.pdf"
    assert result.citations[0]["source"] == "async.pdf"
    assert result.metadata.get("runtime_fallback") is None
    assert result.metadata["async_runtime"]["enabled"] is True
    assert [step.step for step in result.trace] == [
        "planner",
        "tool_call",
        "final_answer",
    ]
