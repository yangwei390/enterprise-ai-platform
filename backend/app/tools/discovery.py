from backend.app.tools.registry import ToolRegistry, get_tool_registry


class ToolDiscoveryService:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def list_available_tools(self) -> list[dict]:
        return [
            descriptor.to_llm_schema()
            for descriptor in self.registry.list_descriptors(enabled_only=True)
        ]

    def refresh(self) -> dict:
        return self.registry.refresh()

    async def arefresh(self) -> dict:
        return await self.registry.arefresh()
