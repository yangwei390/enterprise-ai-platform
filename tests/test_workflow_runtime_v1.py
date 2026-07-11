from backend.app.agents import AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
from backend.app.api.agent import router as agent_router
from backend.app.api.chat import get_chat_service
from backend.app.api.chat import router as chat_router
from backend.app.api.workflow import router as workflow_router
from backend.app.chat import ChatRequest, ChatResponse
from backend.app.tools import ToolResult
from backend.app.workflows import (
    WorkflowRunRequest,
    WorkflowRuntimeV1,
    WorkflowV1Definition,
    WorkflowV1Node,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeToolExecutor:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = []

    def execute(self, tool_call):
        self.calls.append(tool_call)
        if self.should_fail:
            return ToolResult(name=tool_call.name, success=False, error="tool failed")
        if tool_call.name == "knowledge_search":
            return ToolResult(
                name="knowledge_search",
                success=True,
                result={
                    "answer": "第二章讲促进就业。",
                    "sources": [{"source": "中国劳动法.pdf"}],
                    "citations": [{"source": "中国劳动法.pdf"}],
                    "metadata": {"retriever_mode": "hybrid"},
                },
            )
        return ToolResult(name=tool_call.name, success=True, result={"value": 7})


class FakeAgentRuntime:
    def run(self, request):
        return AgentRuntimeResult(
            answer=f"agent: {request.query}",
            action="direct_answer",
            tool_calls=[],
            observations=[],
            sources=[],
            citations=[],
            metadata={},
            trace=[
                AgentTraceStep(
                    step="final_answer",
                    name="final_answer",
                    input={"query": request.query},
                    output={"answer": f"agent: {request.query}"},
                )
            ],
        )


class FakeChatService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            query=request.query,
            answer="chat answer",
            conversation_id=None,
            message_id=None,
            sources=[],
            citations=[],
            context_text="",
            prompt_text="",
            llm_model="fake-llm",
            metadata={},
        )


def test_workflow_runtime_runs_default_knowledge_workflow():
    executor = FakeToolExecutor()
    result = WorkflowRuntimeV1(tool_executor=executor).run(
        WorkflowRunRequest(
            query="劳动法第二章说什么",
            knowledge_base_id=4,
        )
    )

    assert executor.calls[0].name == "knowledge_search"
    assert result.answer == "第二章讲促进就业。"
    assert result.node_outputs["knowledge"]["answer"] == "第二章讲促进就业。"
    assert result.node_outputs["final"]["text"] == "第二章讲促进就业。"


def test_workflow_runtime_resolves_query_variable():
    executor = FakeToolExecutor()

    WorkflowRuntimeV1(tool_executor=executor).run(
        WorkflowRunRequest(query="劳动法第二章说什么", knowledge_base_id=4)
    )

    assert executor.calls[0].arguments["query"] == "劳动法第二章说什么"
    assert executor.calls[0].arguments["knowledge_base_id"] == 4


def test_workflow_runtime_resolves_node_output_variable():
    definition = WorkflowV1Definition(
        id="custom",
        name="custom",
        nodes=[
            WorkflowV1Node(
                id="knowledge",
                type="tool",
                tool_name="knowledge_search",
                input={"query": "{{query}}"},
            ),
            WorkflowV1Node(
                id="final",
                type="echo",
                input={"text": "最终回答：{{knowledge.answer}}"},
            ),
        ],
    )

    result = WorkflowRuntimeV1(tool_executor=FakeToolExecutor()).run(
        WorkflowRunRequest(query="劳动法第二章说什么", definition=definition)
    )

    assert result.answer == "最终回答：第二章讲促进就业。"


def test_workflow_runtime_returns_trace():
    result = WorkflowRuntimeV1(tool_executor=FakeToolExecutor()).run(
        WorkflowRunRequest(query="劳动法第二章说什么", knowledge_base_id=4)
    )

    assert [step.node_id for step in result.trace] == ["knowledge", "final"]
    assert all(step.status == "success" for step in result.trace)
    assert all(step.duration_ms >= 0 for step in result.trace)


def test_workflow_runtime_handles_node_error():
    result = WorkflowRuntimeV1(tool_executor=FakeToolExecutor(should_fail=True)).run(
        WorkflowRunRequest(query="劳动法第二章说什么", knowledge_base_id=4)
    )

    assert result.metadata["failed"] is True
    assert result.trace[0].status == "failed"
    assert result.trace[0].error == "tool failed"
    assert "final" not in result.node_outputs


def test_workflow_run_api_returns_success(monkeypatch):
    class FakeWorkflowRuntime:
        def run(self, request):
            return WorkflowRuntimeV1(tool_executor=FakeToolExecutor()).run(request)

    monkeypatch.setattr(
        "backend.app.api.workflow.WorkflowRuntimeFactory.get_runtime",
        lambda provider=None: FakeWorkflowRuntime(),
    )
    app = FastAPI()
    app.include_router(workflow_router)
    client = TestClient(app)

    response = client.post(
        "/workflow/run",
        json={
            "query": "劳动法第二章说什么",
            "knowledge_base_id": 4,
            "workflow_id": "default_knowledge_workflow",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["data"]["answer"] == "第二章讲促进就业。"
    assert "knowledge" in body["data"]["node_outputs"]


def test_existing_chat_and_agent_api_still_work(monkeypatch):
    class FakeApiAgentRuntime:
        def run(self, request):
            return AgentRuntimeResult(
                answer="agent answer",
                action="direct_answer",
                tool_calls=[],
                observations=[],
                sources=[],
                citations=[],
                metadata={},
                trace=[],
            )

    monkeypatch.setattr("backend.app.api.agent.AgentRuntime", FakeApiAgentRuntime)
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(agent_router)
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    chat_response = client.post("/chat", json={"query": "你好"})
    agent_response = client.post("/agent/chat", json={"query": "你好"})

    assert chat_response.status_code == 200
    assert chat_response.json()["code"] == 0
    assert agent_response.status_code == 200
    assert agent_response.json()["code"] == 0
