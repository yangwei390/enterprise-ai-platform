import json
from pathlib import Path

import pytest
from backend.app.agents.definition import (
    get_agent_definition_registry,
    reset_agent_definition_registry,
)
from backend.app.agents.langgraph.graph import build_agent_graph
from backend.app.agents.langgraph.nodes import ToolNode
from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.api.debug import router as debug_router
from backend.app.config.settings import settings
from backend.app.llms import LLMResponse
from backend.app.tools import BaseTool, ToolCall, ToolExecutor, ToolResult
from backend.app.tools.providers import (
    BuiltinToolProvider,
    HTTPToolProvider,
    PluginToolProvider,
)
from backend.app.tools.providers.workflow import WorkflowToolProvider
from backend.app.tools.registry import ToolDuplicateError, ToolRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class DynamicToolArgs(BaseModel):
    text: str = "hello"


class DynamicTestTool(BaseTool):
    name = "dynamic_test"
    description = "dynamic test tool"
    args_schema = DynamicToolArgs
    source = "test"

    def run(self, arguments: dict) -> ToolResult:
        args = DynamicToolArgs.model_validate(arguments)
        return ToolResult(
            name=self.name,
            success=True,
            result={"text": args.text},
        )


class PluginSelectedLLM:
    def chat(self, request):
        return LLMResponse(
            answer='{"steps":[{"tool":"plugin_echo","args":{"text":"from planner"}}]}',
            model="fake",
        )


class UnknownToolLLM:
    def chat(self, request):
        return LLMResponse(
            answer='{"steps":[{"tool":"missing_tool","args":{}}]}',
            model="fake",
        )


class FakeWorkflowRuntime:
    def run(self, request):
        return type(
            "WorkflowResult",
            (),
            {
                "metadata": {"failed": False},
                "model_dump": lambda self: {"answer": "workflow answer"},
            },
        )()


class FakeDirectLLM:
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


@pytest.fixture(autouse=True)
def reset_agent_definitions():
    reset_agent_definition_registry()
    yield
    reset_agent_definition_registry()


def test_registry_register_and_get_tool():
    registry = ToolRegistry()
    tool = DynamicTestTool()

    registry.register(tool)

    assert registry.get_tool("dynamic_test") is tool
    assert registry.contains("dynamic_test") is True


def test_registry_rejects_duplicate_tool():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())

    with pytest.raises(ToolDuplicateError):
        registry.register(DynamicTestTool())


def test_registry_unregister_tool():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())

    registry.unregister("dynamic_test")

    assert registry.get_tool("dynamic_test") is None


def test_registry_enable_and_disable_tool():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())

    registry.disable("dynamic_test")
    assert registry.get_descriptor("dynamic_test").enabled is False

    registry.enable("dynamic_test")
    assert registry.get_descriptor("dynamic_test").enabled is True


def test_disabled_tool_not_listed_for_llm():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    registry.disable("dynamic_test")

    assert registry.get_tool_definitions() == []
    assert registry.list_descriptors(enabled_only=True) == []


def test_registry_version_changes():
    registry = ToolRegistry()
    version = registry.version

    registry.register(DynamicTestTool())

    assert registry.version > version


def test_builtin_provider_discovers_existing_tools():
    names = {tool.name for tool in BuiltinToolProvider().discover()}

    assert {"calculator", "echo", "get_current_time", "knowledge_search"} <= names


def test_plugin_provider_discovers_tool(tmp_path, monkeypatch):
    plugin_file = _write_plugin(tmp_path)
    monkeypatch.setattr(settings, "TOOL_PLUGIN_ENABLED", True)
    monkeypatch.setattr("backend.app.tools.providers.plugin.PROJECT_ROOT", tmp_path)

    tools = PluginToolProvider(str(plugin_file.parent)).discover()

    assert [tool.name for tool in tools] == ["plugin_echo"]


def test_plugin_failure_does_not_break_other_plugins(tmp_path, monkeypatch):
    _write_plugin(tmp_path)
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')", encoding="utf-8")
    monkeypatch.setattr(settings, "TOOL_PLUGIN_ENABLED", True)
    monkeypatch.setattr("backend.app.tools.providers.plugin.PROJECT_ROOT", tmp_path)
    provider = PluginToolProvider(str(tmp_path))

    tools = provider.discover()

    assert any(tool.name == "plugin_echo" for tool in tools)
    assert provider.errors


def test_http_tool_provider_loads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "http_tools.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "remote_echo",
                    "description": "remote echo",
                    "method": "POST",
                    "url": "http://127.0.0.1:9000/echo",
                    "input_schema": {"type": "object", "properties": {}},
                    "timeout_seconds": 5,
                    "enabled": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "HTTP_TOOL_PROVIDER_ENABLED", True)

    tools = HTTPToolProvider(str(config_path)).discover()

    assert tools[0].name == "remote_echo"


def test_http_tool_uses_async_client(monkeypatch):
    from backend.app.tools.providers.http import HTTPTool

    called = {"post": False}

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = "ok"

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            called["post"] = True
            return FakeResponse()

    monkeypatch.setattr("backend.app.tools.providers.http.httpx.AsyncClient", FakeClient)
    tool = HTTPTool(
        {
            "name": "remote_echo",
            "description": "remote echo",
            "method": "POST",
            "url": "http://127.0.0.1:9000/echo",
        }
    )

    result = __import__("asyncio").run(tool.arun({"text": "hello"}))

    assert called["post"] is True
    assert result.success is True


def test_workflow_provider_registers_workflow_tool():
    tools = WorkflowToolProvider().discover()

    assert tools[0].name == "workflow_default_knowledge"


def test_planner_receives_dynamic_tool_list(monkeypatch):
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    captured = {}

    class CapturingLLM:
        def chat(self, request):
            captured["request"] = request
            return LLMResponse(answer='{"steps":[]}', model="fake")

    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: CapturingLLM(),
    )

    LLMPlanner().plan(query="hello")

    user_payload = json.loads(captured["request"].messages[1].content)
    assert user_payload["available_tools"][0]["function"]["name"] == "dynamic_test"


def test_planner_rejects_unknown_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: UnknownToolLLM(),
    )

    plan = LLMPlanner().plan(query="hello")

    assert plan.steps == []
    assert plan.metadata["tool_registry"]["rejected_tools"] == ["missing_tool"]


def test_planner_cannot_select_disabled_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    registry.disable("dynamic_test")
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )

    plan = LLMPlanner()._normalize_plan(
        plan_data={"steps": [{"tool": "dynamic_test", "args": {}}]},
        query="hello",
        knowledge_base_id=None,
        conversation_id=None,
        memory_context=None,
    )

    assert plan.steps == []


def test_tool_executor_uses_dynamic_registry():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())

    result = ToolExecutor(registry=registry).execute(
        ToolCall(name="dynamic_test", arguments={"text": "hello"})
    )

    assert result.success is True
    assert result.metadata["provider"] == "test"


def test_tool_executor_rejects_disabled_tool():
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    registry.disable("dynamic_test")

    result = ToolExecutor(registry=registry).execute(
        ToolCall(name="dynamic_test", arguments={"text": "hello"})
    )

    assert result.success is False
    assert "disabled" in str(result.error)


def test_provider_refresh_fail_open():
    class FailingProvider:
        @property
        def name(self):
            return "failing"

        def discover(self):
            raise RuntimeError("provider failed")

        def health(self):
            return {"provider": self.name}

    registry = ToolRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FailingProvider())

    result = registry.refresh()

    assert result["failed"] == 1
    assert registry.get_tool("calculator") is not None


def test_debug_tools_api_returns_registry_state(monkeypatch):
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    monkeypatch.setattr("backend.app.api.debug.get_tool_registry", lambda: registry)
    app = FastAPI()
    app.include_router(debug_router)
    client = TestClient(app)

    response = client.get("/debug/tools")

    assert response.status_code == 200
    assert response.json()["data"]["registry_version"] == registry.version


def test_existing_builtin_tools_still_work():
    registry = ToolRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.refresh()

    result = ToolExecutor(registry=registry).execute(
        ToolCall(name="calculator", arguments={"expression": "1 + 2 * 3"})
    )

    assert result.success is True
    assert result.result["value"] == 7


def test_langgraph_agent_selects_dynamic_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(DynamicTestTool())
    definition_registry = get_agent_definition_registry()
    current = definition_registry.get("general_agent")
    definition_registry.register(
        current.model_copy(update={"id": "dynamic_test_agent", "tool_allowlist": ["dynamic_test"]})
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: type(
            "FakeLLM",
            (),
            {
                "chat": lambda self, request: LLMResponse(
                    answer='{"steps":[{"tool":"dynamic_test","args":{"text":"ok"}}]}',
                    model="fake",
                )
            },
        )(),
    )
    graph_app = build_agent_graph(tool_node=ToolNode(tool_executor=ToolExecutor(registry=registry)))

    result = LangGraphAgentRuntime(graph_app=graph_app).run(
        AgentRuntimeRequest(query="use dynamic", agent_id="dynamic_test_agent")
    )

    assert result.tool_calls[0]["name"] == "dynamic_test"
    assert result.answer == "{'text': 'ok'}"


def test_real_dynamic_registry_smoke(tmp_path, monkeypatch):
    plugin_file = _write_plugin(tmp_path)
    registry = ToolRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(PluginToolProvider(str(plugin_file.parent)))
    monkeypatch.setattr(settings, "TOOL_PLUGIN_ENABLED", True)
    monkeypatch.setattr("backend.app.tools.providers.plugin.PROJECT_ROOT", tmp_path)
    refresh_result = registry.refresh()
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: PluginSelectedLLM(),
    )

    plan = LLMPlanner().plan(query="plugin")
    result = ToolExecutor(registry=registry).execute(
        ToolCall(name=plan.steps[0].tool, arguments=plan.steps[0].args)
    )
    registry.unregister("plugin_echo")
    plan_after_unregister = LLMPlanner().plan(query="plugin")

    assert refresh_result["added"] >= 1
    assert plan.steps[0].tool == "plugin_echo"
    assert result.success is True
    assert result.result["text"] == "from planner"
    assert plan_after_unregister.steps == []


def _write_plugin(directory: Path) -> Path:
    plugin_file = directory / "plugin_echo.py"
    plugin_file.write_text(
        """
from backend.app.tools.base import BaseTool, ToolResult
from pydantic import BaseModel

class PluginArgs(BaseModel):
    text: str

class PluginEchoTool(BaseTool):
    name = "plugin_echo"
    description = "Plugin echo"
    args_schema = PluginArgs
    source = "plugin"

    def run(self, arguments):
        args = PluginArgs.model_validate(arguments)
        return ToolResult(name=self.name, success=True, result={"text": args.text})

def create_tool():
    return PluginEchoTool()
""",
        encoding="utf-8",
    )
    return plugin_file
