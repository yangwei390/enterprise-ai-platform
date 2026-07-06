from backend.app.tools.base import BaseTool, ToolDefinition
from backend.app.tools.builtin_tools import get_builtin_tools


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [tool.get_definition() for tool in self.list_tools()]


_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        for tool in get_builtin_tools():
            _tool_registry.register(tool)
    return _tool_registry
