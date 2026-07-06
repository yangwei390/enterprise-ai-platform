from backend.app.tools.base import ToolCall, ToolResult
from backend.app.tools.registry import ToolRegistry, get_tool_registry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self.registry.get_tool(tool_call.name)
        if tool is None:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error="tool not found",
            )

        try:
            return tool.run(tool_call.arguments)
        except Exception as exc:
            return ToolResult(name=tool_call.name, success=False, error=str(exc))
