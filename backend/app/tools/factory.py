from backend.app.tools.discovery import ToolDiscoveryService
from backend.app.tools.registry import ToolRegistry, get_tool_registry


class ToolRegistryFactory:
    @staticmethod
    def get_registry() -> ToolRegistry:
        return get_tool_registry()

    @staticmethod
    def get_discovery_service() -> ToolDiscoveryService:
        return ToolDiscoveryService(get_tool_registry())
