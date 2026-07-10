import asyncio

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.mcp.client_manager import (
    MCPClientManager,
    get_mcp_client_manager,
)
from backend.app.mcp.tools import MCPToolAdapter
from backend.app.tools.base import BaseTool
from backend.app.tools.providers.base import BaseToolProvider


class MCPToolProvider(BaseToolProvider):
    def __init__(self, manager: MCPClientManager | None = None) -> None:
        self.manager = manager
        self.last_result: dict = {}
        self.errors: list[dict] = []

    @property
    def name(self) -> str:
        return "mcp"

    def discover(self) -> list[BaseTool]:
        if not settings.MCP_ENABLED and not settings.MCP_TOOL_PROVIDER_ENABLED:
            self.last_result = {"enabled": False, "discovered_tools": 0}
            return []
        try:
            return asyncio.run(self.adiscover())
        except Exception as exc:
            self.errors.append({"provider": self.name, "error": str(exc)})
            self.last_result = {
                "enabled": True,
                "discovered_tools": 0,
                "errors": self.errors,
            }
            logger.exception("MCP provider discovery failed")
            if settings.MCP_FAIL_OPEN:
                return []
            raise

    async def adiscover(self) -> list[BaseTool]:
        if not settings.MCP_ENABLED and not settings.MCP_TOOL_PROVIDER_ENABLED:
            self.last_result = {"enabled": False, "discovered_tools": 0}
            return []

        manager = self.manager or get_mcp_client_manager()
        discovered, metadata = await manager.discover_all_tools()
        self.last_result = metadata
        self.errors = metadata.get("errors", [])
        tools: list[BaseTool] = []
        for server_config, remote_tool in discovered:
            tools.append(
                MCPToolAdapter(
                    server_config=server_config,
                    remote_tool=remote_tool,
                    manager=manager,
                )
            )
        return tools

    def health(self) -> dict:
        return {
            "provider": self.name,
            "healthy": not self.errors,
            "implemented": True,
            "enabled": settings.MCP_ENABLED or settings.MCP_TOOL_PROVIDER_ENABLED,
            "last_result": self.last_result,
            "errors": self.errors,
        }
