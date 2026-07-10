from backend.app.mcp.client import MCPClient
from backend.app.mcp.client_manager import MCPClientManager, get_mcp_client_manager
from backend.app.mcp.schemas import MCPHealthStatus, MCPServerConfig

__all__ = [
    "MCPClient",
    "MCPClientManager",
    "MCPHealthStatus",
    "MCPServerConfig",
    "get_mcp_client_manager",
]


def __getattr__(name: str):
    if name == "MCPAdapter":
        from backend.app.mcp.adapter import MCPAdapter

        return MCPAdapter
    if name in {"MCPToolCallRequest", "MCPToolCallResult", "MCPToolConfig"}:
        from backend.app.mcp import base

        return getattr(base, name)
    if name == "RemoteHTTPTool":
        from backend.app.mcp.remote_tool import RemoteHTTPTool

        return RemoteHTTPTool
    raise AttributeError(name)
