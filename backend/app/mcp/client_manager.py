import asyncio
from time import perf_counter
from typing import Any, cast

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.mcp.client import MCPClient
from backend.app.mcp.config import load_mcp_server_configs
from backend.app.mcp.schemas import MCPHealthStatus, MCPServerConfig
from backend.app.mcp.security import hash_arguments


class MCPClientManager:
    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self.configs = {config.name: config for config in (configs or [])}
        if configs is None:
            self.configs = {
                config.name: config
                for config in load_mcp_server_configs()
                if config.enabled
            }
        self.clients: dict[str, MCPClient] = {
            name: MCPClient(config) for name, config in self.configs.items()
        }
        self.audit_records: list[dict] = []
        self._semaphore = asyncio.Semaphore(settings.MCP_MAX_CONCURRENCY)

    async def connect(self, server_name: str) -> None:
        client = self._get_client(server_name)
        async with self._semaphore:
            await client.connect()

    async def disconnect(self, server_name: str) -> None:
        client = self._get_client(server_name)
        await client.disconnect()

    async def connect_all(self) -> dict:
        results = await asyncio.gather(
            *[self._connect_one(name) for name in self.clients],
            return_exceptions=False,
        )
        return {"servers": results}

    async def disconnect_all(self) -> None:
        for client in self.clients.values():
            try:
                await client.disconnect()
            except BaseException as exc:
                logger.debug(
                    f"MCP client disconnect ignored cleanup exception | "
                    f"server={client.config.name} | error={exc}"
                )

    def list_servers(self) -> list[dict]:
        return [
            {
                "name": config.name,
                "enabled": config.enabled,
                "transport": config.transport,
                "tags": config.tags,
                "metadata": config.metadata,
            }
            for config in self.configs.values()
        ]

    async def health(self, server_name: str) -> dict:
        return (await self._get_client(server_name).health()).model_dump()

    async def health_all(self) -> dict:
        statuses = await asyncio.gather(
            *[client.health() for client in self.clients.values()],
            return_exceptions=True,
        )
        return {
            "servers": [
                status.model_dump()
                for status in statuses
                if isinstance(status, MCPHealthStatus)
            ]
        }

    async def discover_tools(self, server_name: str) -> list[tuple[MCPServerConfig, Any]]:
        client = self._get_client(server_name)
        await client.connect()
        result = await client.list_tools()
        tools = getattr(result, "tools", [])
        return [(client.config, tool) for tool in tools]

    async def discover_all_tools(self) -> tuple[list[tuple[MCPServerConfig, Any]], dict]:
        started_at = perf_counter()
        discovered: list[tuple[MCPServerConfig, Any]] = []
        connected_servers: list[str] = []
        failed_servers: list[dict] = []

        results = await asyncio.gather(
            *[self._discover_one(name) for name in self.clients],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                failed_servers.append({"server": "unknown", "error": str(result)})
                continue
            server_name, tools, error = cast(
                tuple[str, list[tuple[MCPServerConfig, Any]], str | None],
                result,
            )
            if error:
                failed_servers.append({"server": server_name, "error": error})
                continue
            connected_servers.append(server_name)
            discovered.extend(tools)

        metadata = {
            "connected_servers": connected_servers,
            "failed_servers": failed_servers,
            "discovered_tools": len(discovered),
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "errors": failed_servers,
        }
        return discovered, metadata

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> Any:
        client = self._get_client(server_name)
        started_at = perf_counter()
        audit = {
            "server_name": server_name,
            "remote_tool_name": tool_name,
            "transport": client.config.transport,
            "arguments_hash": hash_arguments(arguments),
            "status": "running",
            "retry_count": 0,
            "timeout": False,
            "error_code": None,
        }
        try:
            async with self._semaphore:
                result = await client.call_tool(tool_name, arguments)
            audit["status"] = "success"
            return result
        except Exception as exc:
            audit["status"] = "failed"
            audit["error_code"] = type(exc).__name__
            logger.exception(
                f"MCP tool call failed | server={server_name} | tool={tool_name}"
            )
            raise
        finally:
            audit["duration_ms"] = round((perf_counter() - started_at) * 1000, 2)
            self.audit_records.append(audit)

    def _get_client(self, server_name: str) -> MCPClient:
        client = self.clients.get(server_name)
        if client is None:
            raise KeyError(f"MCP server not found: {server_name}")
        return client

    async def _connect_one(self, server_name: str) -> dict:
        try:
            await self.connect(server_name)
            return {"server": server_name, "connected": True}
        except Exception as exc:
            return {"server": server_name, "connected": False, "error": str(exc)}

    async def _discover_one(
        self, server_name: str
    ) -> tuple[str, list[tuple[MCPServerConfig, Any]], str | None]:
        try:
            tools = await self.discover_tools(server_name)
            return server_name, tools, None
        except Exception as exc:
            logger.exception(f"MCP tool discovery failed | server={server_name}")
            return server_name, [], str(exc)


_mcp_client_manager: MCPClientManager | None = None


def get_mcp_client_manager(*, reload: bool = False) -> MCPClientManager:
    global _mcp_client_manager
    if _mcp_client_manager is None or reload:
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager


def set_mcp_client_manager(manager: MCPClientManager | None) -> None:
    global _mcp_client_manager
    _mcp_client_manager = manager
