import asyncio
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.api.debug import router as debug_router
from backend.app.config.settings import settings
from backend.app.llms import LLMResponse
from backend.app.mcp.client_manager import MCPClientManager
from backend.app.mcp.config import load_mcp_server_configs
from backend.app.mcp.errors import MCPConfigError, MCPPermissionError
from backend.app.mcp.schemas import MCPServerConfig
from backend.app.mcp.security import redact_mapping
from backend.app.tools import BaseTool, ToolCall, ToolExecutor
from backend.app.tools.providers import BuiltinToolProvider, MCPToolProvider
from backend.app.tools.registry import ToolRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

DEMO_SERVER = "tests/fixtures/mcp_demo_server.py"


def test_mcp_config_loads_stdio_server(tmp_path):
    config_path = tmp_path / "servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "local_demo",
                        "transport": "stdio",
                        "command": "python3.12",
                        "args": [DEMO_SERVER],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(str(config_path))

    assert configs[0].name == "local_demo"
    assert configs[0].transport == "stdio"


def test_mcp_config_loads_streamable_http_server(tmp_path):
    config_path = tmp_path / "servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "remote_demo",
                        "transport": "streamable_http",
                        "url": "http://127.0.0.1:9000/mcp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(str(config_path))

    assert configs[0].transport == "streamable_http"


def test_mcp_env_header_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret-token")
    config_path = tmp_path / "servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "remote_demo",
                        "transport": "streamable_http",
                        "url": "http://127.0.0.1:9000/mcp",
                        "headers": {"Authorization": "${MCP_TEST_TOKEN}"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(str(config_path))

    assert configs[0].headers["Authorization"] == "secret-token"


def test_mcp_sensitive_headers_are_redacted():
    redacted = redact_mapping({"Authorization": "Bearer abc", "X-Trace": "1"})

    assert redacted["Authorization"] == "***"
    assert redacted["X-Trace"] == "1"


def test_stdio_transport_rejects_disallowed_command():
    with pytest.raises(MCPPermissionError):
        load_config_from_dict(
            {
                "servers": [
                    {
                        "name": "bad",
                        "transport": "stdio",
                        "command": "bash",
                    }
                ]
            }
        )


def test_streamable_http_rejects_invalid_url():
    with pytest.raises(MCPConfigError):
        load_config_from_dict(
            {
                "servers": [
                    {
                        "name": "bad",
                        "transport": "streamable_http",
                        "url": "file:///tmp/mcp",
                    }
                ]
            }
        )


def test_mcp_client_connect_and_initialize():
    async def run() -> dict:
        manager = build_stdio_manager()
        try:
            await manager.connect("local_demo")
            return await manager.health("local_demo")
        finally:
            await manager.disconnect_all()

    health = asyncio.run(run())

    assert health["connected"] is True
    assert health["healthy"] is True


def test_mcp_client_lists_tools():
    async def run() -> list[str]:
        manager = build_stdio_manager()
        try:
            tools = await manager.discover_tools("local_demo")
            return [tool.name for _, tool in tools]
        finally:
            await manager.disconnect_all()

    assert {"mcp_echo", "mcp_add"} <= set(asyncio.run(run()))


def test_mcp_client_calls_tool():
    async def run() -> dict:
        manager = build_stdio_manager()
        try:
            result = await manager.call_tool("local_demo", "mcp_add", {"a": 1, "b": 2})
            return result.model_dump(by_alias=True)
        finally:
            await manager.disconnect_all()

    result = asyncio.run(run())

    assert result["isError"] is False
    assert "3" in result["content"][0]["text"]


def test_mcp_client_disconnect_is_idempotent():
    async def run() -> None:
        manager = build_stdio_manager()
        await manager.disconnect_all()
        await manager.disconnect_all()

    asyncio.run(run())


def test_mcp_client_manager_handles_multiple_servers():
    manager = MCPClientManager(
        [
            stdio_config("local_demo_one"),
            stdio_config("local_demo_two"),
        ]
    )

    assert {server["name"] for server in manager.list_servers()} == {
        "local_demo_one",
        "local_demo_two",
    }


def test_one_failed_server_does_not_break_other_servers():
    async def run() -> dict:
        manager = MCPClientManager(
            [
                stdio_config("local_demo"),
                MCPServerConfig(
                    name="bad_demo",
                    transport="stdio",
                    command="python3.12",
                    args=["missing_mcp_server.py"],
                    connect_timeout_seconds=1,
                    timeout_seconds=1,
                ),
            ]
        )
        try:
            _tools, metadata = await manager.discover_all_tools()
            return metadata
        finally:
            await manager.disconnect_all()

    metadata = asyncio.run(run())

    assert "local_demo" in metadata["connected_servers"]
    assert metadata["failed_servers"]


def test_mcp_provider_discovers_tools(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    provider = MCPToolProvider(build_stdio_manager())

    tools = provider.discover()

    assert {tool.name for tool in tools} == {
        "mcp__local_demo__mcp_echo",
        "mcp__local_demo__mcp_add",
    }


def test_mcp_tool_descriptor_schema(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    tool = MCPToolProvider(build_stdio_manager()).discover()[0]
    descriptor = tool.get_descriptor()

    assert descriptor.provider == "mcp"
    assert descriptor.async_supported is True
    assert descriptor.metadata["mcp_server"] == "local_demo"


def test_mcp_tool_registered_in_dynamic_registry(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    registry = ToolRegistry()
    registry.register_provider(MCPToolProvider(build_stdio_manager()))
    result = registry.refresh()

    assert result["added"] >= 2
    assert registry.get_tool("mcp__local_demo__mcp_add") is not None


def test_planner_receives_mcp_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(DummyDescriptorTool("mcp__local_demo__mcp_add"))
    captured: dict[str, Any] = {}

    class FakeLLM:
        def chat(self, request):
            captured["request"] = request
            return LLMResponse(answer='{"steps":[]}', model="fake")

    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: FakeLLM(),
    )

    LLMPlanner().plan(query="请调用 mcp_add")

    content = captured["request"].messages[1].content
    assert "mcp__local_demo__mcp_add" in content


def test_tool_executor_executes_mcp_tool(monkeypatch):
    async def run() -> bool:
        monkeypatch.setattr(settings, "MCP_ENABLED", True)
        manager = build_stdio_manager()
        provider = MCPToolProvider(manager)
        registry = ToolRegistry()
        for tool in await provider.adiscover():
            registry.register(tool, replace=True)
        try:
            result = await ToolExecutor(registry).aexecute(
                ToolCall(name="mcp__local_demo__mcp_add", arguments={"a": 2, "b": 5})
            )
            return result.success and "7" in str(result.result)
        finally:
            await manager.disconnect_all()

    assert asyncio.run(run()) is True


def test_mcp_tool_timeout(monkeypatch):
    async def run() -> str | None:
        monkeypatch.setattr(settings, "MCP_ENABLED", True)
        manager = build_stdio_manager()
        provider = MCPToolProvider(manager)
        registry = ToolRegistry()
        for tool in await provider.adiscover():
            registry.register(tool, replace=True)
        try:
            monkeypatch.setattr(settings, "AGENT_TOOL_TIMEOUT_SECONDS", 0)
            result = await ToolExecutor(registry).aexecute(
                ToolCall(name="mcp__local_demo__mcp_add", arguments={"a": 2, "b": 5})
            )
            return result.error
        finally:
            await manager.disconnect_all()

    assert "timed out" in str(asyncio.run(run()))


def test_mcp_tool_retry(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    monkeypatch.setattr(settings, "AGENT_TOOL_RETRY_COUNT", 1)
    registry = ToolRegistry()
    registry.register(FailsOnceTool())

    result = asyncio.run(
        ToolExecutor(registry).aexecute(
            ToolCall(name="mcp__fake__retry", arguments={})
        )
    )

    assert result.success is True
    assert result.metadata["attempt_count"] == 2


def test_mcp_permission_denies_tool(monkeypatch):
    monkeypatch.setattr(settings, "MCP_PERMISSION_ENFORCEMENT_ENABLED", True)
    config = stdio_config("local_demo").model_copy(
        update={"required_permissions": ["admin:mcp"]}
    )

    async def run() -> bool:
        manager = MCPClientManager([config])
        provider = MCPToolProvider(manager)
        registry = ToolRegistry()
        for tool in await provider.adiscover():
            registry.register(tool, replace=True)
        try:
            result = await ToolExecutor(registry).aexecute(
                ToolCall(name="mcp__local_demo__mcp_add", arguments={"a": 1, "b": 1})
            )
            return result.success
        finally:
            await manager.disconnect_all()

    assert asyncio.run(run()) is False


def test_mcp_audit_redacts_secrets():
    assert redact_mapping({"Cookie": "abc", "Safe": "ok"}) == {
        "Cookie": "***",
        "Safe": "ok",
    }


def test_debug_mcp_api(monkeypatch):
    app = FastAPI()
    app.include_router(debug_router)
    monkeypatch.setattr(settings, "MCP_ENABLED", False)

    response = TestClient(app).get("/debug/mcp")

    assert response.status_code == 200
    assert "configured_servers" in response.json()["data"]


def test_fastapi_shutdown_closes_mcp_clients():
    async def run() -> None:
        manager = build_stdio_manager()
        await manager.connect("local_demo")
        await manager.disconnect_all()
        assert (await manager.health("local_demo"))["connected"] is False

    asyncio.run(run())


def test_v1_tools_still_work():
    names = {tool.name for tool in BuiltinToolProvider().discover()}

    assert {"calculator", "echo", "get_current_time", "knowledge_search"} <= names


def test_langgraph_agent_selects_mcp_tool(monkeypatch):
    registry = ToolRegistry()
    registry.register(DummyDescriptorTool("mcp__local_demo__mcp_add"))

    class FakeLLM:
        def chat(self, request):
            return LLMResponse(
                answer='{"steps":[{"tool":"mcp__local_demo__mcp_add","args":{"a":12,"b":30}}]}',
                model="fake",
            )

    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: FakeLLM(),
    )

    plan = LLMPlanner().plan(query="请调用 mcp_add 计算 12 加 30")

    assert plan.steps[0].tool == "mcp__local_demo__mcp_add"


def test_real_mcp_stdio_smoke(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", True)

    async def run() -> bool:
        manager = build_stdio_manager()
        provider = MCPToolProvider(manager)
        registry = ToolRegistry()
        for tool in await provider.adiscover():
            registry.register(tool, replace=True)
        try:
            planner_registry = registry
            plan = LLMPlanner()._normalize_plan(
                plan_data={
                    "steps": [
                        {
                            "tool": "mcp__local_demo__mcp_add",
                            "args": {"a": 12, "b": 30},
                        }
                    ]
                },
                tool_descriptors={
                    descriptor.name: descriptor
                    for descriptor in planner_registry.list_descriptors(enabled_only=True)
                },
                query="请调用 mcp_add 计算 12 加 30",
                knowledge_base_id=None,
                conversation_id=None,
                memory_context=None,
            )
            result = await ToolExecutor(registry).aexecute(
                ToolCall(name=plan.steps[0].tool, arguments=plan.steps[0].args)
            )
            return result.success and "42" in str(result.result)
        finally:
            await manager.disconnect_all()

    assert asyncio.run(run()) is True


def test_real_mcp_streamable_http_smoke():
    port = _free_port_or_skip()
    proc = subprocess.Popen(
        ["python3.12", DEMO_SERVER, "streamable-http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)

        async def run() -> bool:
            manager = MCPClientManager(
                [
                    MCPServerConfig(
                        name="http_demo",
                        transport="streamable_http",
                        url=f"http://127.0.0.1:{port}/mcp",
                        timeout_seconds=10,
                        connect_timeout_seconds=10,
                        tool_call_timeout_seconds=10,
                    )
                ]
            )
            try:
                tools = await manager.discover_tools("http_demo")
                result = await manager.call_tool(
                    "http_demo", "mcp_add", {"a": 5, "b": 6}
                )
                return (
                    {"mcp_echo", "mcp_add"} <= {tool.name for _, tool in tools}
                    and "11" in str(result.model_dump(by_alias=True))
                )
            finally:
                await manager.disconnect_all()

        assert asyncio.run(run()) is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def load_config_from_dict(data: dict) -> list[MCPServerConfig]:
    path = Path("/tmp/mcp_test_config.json")
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_mcp_server_configs(str(path))


def stdio_config(name: str = "local_demo") -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command="python3.12",
        args=[DEMO_SERVER],
        cwd=".",
        timeout_seconds=10,
        connect_timeout_seconds=10,
        tool_call_timeout_seconds=10,
    )


def build_stdio_manager() -> MCPClientManager:
    return MCPClientManager([stdio_config()])


def _free_port_or_skip() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        return int(port)
    except PermissionError:
        pytest.skip("local port binding is not permitted in this sandbox")
    finally:
        sock.close()


class EmptyArgs(BaseModel):
    pass


class DummyDescriptorTool(BaseTool):
    args_schema = EmptyArgs

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "dummy mcp descriptor"
        self.source = "mcp"
        self.permission = "public"

    def run(self, arguments: dict):
        raise NotImplementedError


class FailsOnceTool(BaseTool):
    name = "mcp__fake__retry"
    description = "fails once"
    source = "mcp"
    permission = "public"
    calls = 0
    args_schema = EmptyArgs

    def run(self, arguments: dict):
        raise NotImplementedError

    async def arun(self, arguments: dict):
        from backend.app.tools import ToolResult

        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return ToolResult(name=self.name, success=True, result={"ok": True})
