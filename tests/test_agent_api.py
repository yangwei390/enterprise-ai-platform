from backend.app.agents import AgentRuntimeResult
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.trace import AgentTraceStep
from backend.app.api.agent import get_conversation_service
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


class FakeStreamingAgentRuntime:
    async def astream_events(self, request):
        yield {"event": "status", "data": {"status": "answering", "message": "正在整理答案"}}
        yield {"event": "answer_delta", "data": {"delta": "流式"}}
        yield {"event": "answer_delta", "data": {"delta": "回答"}}
        yield {
            "event": "result",
            "data": {
                "result": AgentRuntimeResult(
                    answer="流式回答",
                    action="direct_answer",
                    sources=[{"source": "source.pdf"}],
                    citations=[{"source": "source.pdf"}],
                    metadata={"answer_stream_delta_count": 2},
                ).model_dump()
            },
        }


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


class FakeConversation:
    def __init__(self, id: int) -> None:
        self.id = id


class FakeMessage:
    def __init__(self, id: int, content: str, metadata: dict | None = None) -> None:
        self.id = id
        self.content = content
        self.message_metadata = metadata or {}


class FakeConversationService:
    def __init__(self) -> None:
        self.assistant_messages: list[FakeMessage] = []

    def create_conversation(self, data):
        return FakeConversation(901)

    def get_conversation(self, id: int):
        return FakeConversation(id)

    def add_user_message(self, conversation_id: int, content: str, metadata=None):
        return FakeMessage(1, content, metadata)

    def add_assistant_message(self, conversation_id: int, content: str, metadata=None):
        message = FakeMessage(2, content, metadata)
        self.assistant_messages.append(message)
        return message


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


def _parse_sse_events(text: str) -> list[dict]:
    events = []
    for raw_event in text.strip().split("\n\n"):
        event_name = "message"
        data = "{}"
        for line in raw_event.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            if line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        import json

        events.append({"event": event_name, "data": json.loads(data)})
    return events


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


def test_agent_chat_api_enters_langgraph_runtime(monkeypatch):
    called = {}

    async def fake_arun(self, request):
        called["runtime"] = self
        called["request"] = request
        return AgentRuntimeResult(
            answer="v2 answer",
            action="direct_answer",
            metadata={"runtime": "langgraph_v2"},
        )

    monkeypatch.setattr(LangGraphAgentRuntime, "arun", fake_arun)
    app = FastAPI()
    app.include_router(agent_router)
    client = TestClient(app)

    response = client.post("/agent/chat", json={"query": "你好"})

    body = response.json()
    assert response.status_code == 200
    assert body["data"]["answer"] == "v2 answer"
    assert body["data"]["metadata"]["runtime"] == "langgraph_v2"
    assert isinstance(called["runtime"], LangGraphAgentRuntime)
    assert called["request"].query == "你好"


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


def test_agent_stream_api_returns_multiple_answer_deltas(monkeypatch):
    service = FakeConversationService()
    monkeypatch.setattr(
        "backend.app.api.agent.AgentRuntimeFactory.get_runtime",
        lambda: FakeStreamingAgentRuntime(),
    )
    app = FastAPI()
    app.include_router(agent_router)
    app.dependency_overrides[get_conversation_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/agent/chat/stream",
        json={"agent_id": "general_agent", "query": "你好"},
    )
    events = _parse_sse_events(response.text)

    event_names = [event["event"] for event in events]
    deltas = [event["data"]["delta"] for event in events if event["event"] == "answer_delta"]
    completed = next(event for event in events if event["event"] == "completed")

    assert response.status_code == 200
    assert "message_start" in event_names
    assert "status" in event_names
    assert deltas == ["流式", "回答"]
    assert "".join(deltas) == "流式回答"
    assert completed["data"]["answer"] == "流式回答"
    assert service.assistant_messages[0].content == "流式回答"
    assert service.assistant_messages[0].message_metadata["citations"] == [{"source": "source.pdf"}]


def test_agent_stream_api_enters_langgraph_runtime(monkeypatch):
    service = FakeConversationService()
    called = {}

    async def fake_astream_events(self, request):
        called["runtime"] = self
        called["request"] = request
        yield {"event": "status", "data": {"status": "answering", "message": "正在整理答案"}}
        yield {
            "event": "result",
            "data": {
                "result": AgentRuntimeResult(
                    answer="v2 stream answer",
                    action="direct_answer",
                    metadata={"runtime": "langgraph_v2"},
                ).model_dump()
            },
        }

    monkeypatch.setattr(LangGraphAgentRuntime, "astream_events", fake_astream_events)
    app = FastAPI()
    app.include_router(agent_router)
    app.dependency_overrides[get_conversation_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/agent/chat/stream",
        json={"agent_id": "general_agent", "query": "你好"},
    )
    events = _parse_sse_events(response.text)
    completed = next(event for event in events if event["event"] == "completed")

    assert response.status_code == 200
    assert completed["data"]["answer"] == "v2 stream answer"
    assert isinstance(called["runtime"], LangGraphAgentRuntime)
    assert called["request"].query == "你好"
    assert called["request"].metadata["agent_id"] == "general_agent"
