from backend.app.mcp.base import MCPToolConfig
from backend.app.mcp.remote_tool import RemoteHTTPTool
from backend.app.tools import BaseTool, ToolRegistry, get_tool_registry


class MCPAdapter:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def register_remote_tool(self, config: MCPToolConfig) -> BaseTool:
        tool = RemoteHTTPTool(config)
        self.registry.register(tool)
        return tool

    def register_remote_tools(self, configs: list[MCPToolConfig]) -> list[BaseTool]:
        return [self.register_remote_tool(config) for config in configs]
