from backend.app.tools.base import BaseTool
from backend.app.tools.providers.base import BaseToolProvider


class MCPToolProvider(BaseToolProvider):
    @property
    def name(self) -> str:
        return "mcp"

    def discover(self) -> list[BaseTool]:
        return []

    def health(self) -> dict:
        return {
            "provider": self.name,
            "healthy": True,
            "implemented": False,
            "reason": "MCP protocol integration is reserved for a later sprint.",
        }
