import asyncio
import time

from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.memory.factory import MemoryFactory
from backend.app.memory.manager import CheckpointManager, MemoryManager
from backend.app.memory.providers import InMemoryMemoryProvider, RedisMemoryProvider
from backend.app.memory.state import MemoryState
from backend.app.tools import BaseTool, ToolCall, ToolExecutor, ToolResult
from backend.app.tools.registry import ToolRegistry
from pydantic import BaseModel


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, tuple[str, float | None]] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = (value, time.time() + ttl)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = (value, time.time() + ex if ex else None)

    def get(self, key: str) -> str | None:
        item = self.store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and expires_at < time.time():
            self.store.pop(key, None)
            return None
        return value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def scan_iter(self, pattern: str):
        prefix = pattern[:-1] if pattern.endswith("*") else pattern
        for key in list(self.store):
            if key.startswith(prefix) and self.get(key) is not None:
                yield key


class CacheArgs(BaseModel):
    text: str


class CacheableTool(BaseTool):
    name = "cacheable"
    description = "cacheable tool"
    args_schema = CacheArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(
            name=self.name,
            success=True,
            result={"text": arguments["text"], "calls": self.calls},
        )


class SessionGraph:
    def invoke(self, state):
        state["messages"].append({"role": "assistant", "content": "answer"})
        state["final_answer"] = "session answer"
        state["metadata"]["trace"].append(
            {
                "step": "final_answer",
                "name": "final_answer",
                "input": {},
                "output": {"answer": "session answer"},
                "duration_ms": 0,
                "status": "success",
                "error": None,
            }
        )
        return state

    async def ainvoke(self, state):
        return self.invoke(state)


def test_redis_provider_session_save_and_load():
    provider = RedisMemoryProvider(redis_client=FakeRedis(), prefix="test")
    state = MemoryState(session_id="s1", messages=[{"role": "user", "content": "hi"}])

    provider.save_session(state, ttl_seconds=60)
    loaded = provider.load_session("s1")

    assert loaded is not None
    assert loaded.messages[0]["content"] == "hi"


def test_session_save_and_load():
    manager = MemoryManager(InMemoryMemoryProvider())
    state = MemoryState(session_id="s1", current_step="planner")

    manager.save_session(state)

    assert manager.load_session("s1").current_step == "planner"


def test_tool_cache_miss_then_hit(monkeypatch):
    manager = MemoryManager(InMemoryMemoryProvider())
    monkeypatch.setattr(MemoryFactory, "get_manager", staticmethod(lambda provider=None: manager))
    tool = CacheableTool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    first = executor.execute(ToolCall(name="cacheable", arguments={"text": "hello"}))
    second = executor.execute(ToolCall(name="cacheable", arguments={"text": "hello"}))

    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert tool.calls == 1


def test_checkpoint_save_and_load():
    manager = MemoryManager(InMemoryMemoryProvider())
    checkpoint = CheckpointManager(manager)

    checkpoint.save("cp1", {"state": "ok"})

    assert checkpoint.load("cp1") == {"state": "ok"}
    assert checkpoint.list() == ["cp1"]


def test_memory_factory_returns_memory_provider(monkeypatch):
    monkeypatch.setattr("backend.app.memory.factory.settings.MEMORY_PROVIDER", "memory")

    manager = MemoryFactory.get_manager()

    assert manager.provider.name == "memory"


def test_langgraph_memory_injection(monkeypatch):
    manager = MemoryManager(InMemoryMemoryProvider())
    manager.save_session(
        MemoryState(
            session_id="session-1",
            messages=[{"role": "user", "content": "previous"}],
            tool_results=[{"name": "tool"}],
            trace_id="trace-1",
        )
    )
    monkeypatch.setattr(MemoryFactory, "get_manager", staticmethod(lambda provider=None: manager))

    result = LangGraphAgentRuntime(graph_app=SessionGraph()).run(
        AgentRuntimeRequest(query="next", metadata={"session_id": "session-1"})
    )
    saved = manager.load_session("session-1")

    assert result.metadata["session"]["loaded"] is True
    assert result.metadata["session"]["restored_tool_result_count"] == 1
    assert saved is not None
    assert saved.messages[0]["content"] == "previous"


def test_agent_session_restore(monkeypatch):
    manager = MemoryManager(InMemoryMemoryProvider())
    monkeypatch.setattr(MemoryFactory, "get_manager", staticmethod(lambda provider=None: manager))
    runtime = LangGraphAgentRuntime(graph_app=SessionGraph())

    runtime.run(AgentRuntimeRequest(query="hello", metadata={"session_id": "same"}))
    result = runtime.run(AgentRuntimeRequest(query="again", metadata={"session_id": "same"}))

    assert result.metadata["session"]["loaded"] is True


def test_tool_cache_ttl():
    provider = InMemoryMemoryProvider()
    provider.set_cache("key", {"value": 1}, ttl_seconds=1)

    assert provider.get_cache("key") == {"value": 1}
    provider.cache["key"] = ({"value": 1}, time.time() - 1)

    assert provider.get_cache("key") is None


def test_async_tool_cache_hit(monkeypatch):
    manager = MemoryManager(InMemoryMemoryProvider())
    monkeypatch.setattr(MemoryFactory, "get_manager", staticmethod(lambda provider=None: manager))
    tool = CacheableTool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    first = asyncio.run(
        executor.aexecute(ToolCall(name="cacheable", arguments={"text": "hello"}))
    )
    second = asyncio.run(
        executor.aexecute(ToolCall(name="cacheable", arguments={"text": "hello"}))
    )

    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert tool.calls == 1
