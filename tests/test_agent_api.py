from backend.app.agents import AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
from backend.app.api.agent import router as agent_router
from backend.app.api.chat import get_chat_service
from backend.app.api.chat import router as chat_router
from backend.app.chat import ChatRequest, ChatResponse
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeAgentRuntime:
    called = False
    last_request = None

    def run(self, request):
        FakeAgentRuntime.called = True
        FakeAgentRuntime.last_request = request
        return AgentRuntimeResult(
            answer="agent answer",
            action="tool",
            tool_calls=[
                {
                    "name": "knowledge_search",
                    "arguments": {"query": request.query},
                }
            ],
            observations=[{"success": True}],
            sources=[{"source": "source.pdf"}],
            citations=[{"source": "source.pdf"}],
            metadata={"runtime": "fake"},
            trace=[
                AgentTraceStep(
                    step="planner",
                    name="simple_planner",
                    input={"query": request.query},
                    output={"action": "tool"},
                ),
                AgentTraceStep(
                    step="final_answer",
                    name="final_answer",
                    input={"query": request.query},
                    output={"answer": "agent answer"},
                ),
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


def _agent_client(monkeypatch) -> TestClient:
    FakeAgentRuntime.called = False
    FakeAgentRuntime.last_request = None
    monkeypatch.setattr(
        "backend.app.api.agent.AgentRuntimeFactory.get_runtime",
        lambda: FakeAgentRuntime(),
    )
    app = FastAPI()
    app.include_router(agent_router)
    return TestClient(app)


def test_agent_chat_api_returns_success(monkeypatch):
    client = _agent_client(monkeypatch)

    response = client.post(
        "/agent/chat",
        json={
            "query": "劳动法第二章说什么",
            "knowledge_base_id": 4,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["data"]["answer"] == "agent answer"
    assert body["data"]["sources"][0]["source"] == "source.pdf"


def test_agent_chat_api_returns_trace(monkeypatch):
    client = _agent_client(monkeypatch)

    response = client.post("/agent/chat", json={"query": "现在几点"})

    trace_steps = [step["step"] for step in response.json()["data"]["trace"]]
    assert "planner" in trace_steps
    assert "final_answer" in trace_steps


def test_agent_chat_api_uses_agent_runtime(monkeypatch):
    client = _agent_client(monkeypatch)

    response = client.post(
        "/agent/chat",
        json={
            "query": "1+2等于多少",
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    assert FakeAgentRuntime.called is True
    assert FakeAgentRuntime.last_request is not None
    assert FakeAgentRuntime.last_request.query == "1+2等于多少"
    assert FakeAgentRuntime.last_request.metadata == {"source": "test"}


def test_agent_chat_api_rejects_empty_query(monkeypatch):
    client = _agent_client(monkeypatch)

    response = client.post("/agent/chat", json={"query": "   "})

    assert response.status_code == 422
    assert FakeAgentRuntime.called is False


def test_chat_api_still_works():
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/chat", json={"query": "你好"})

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["data"]["answer"] == "chat answer"
