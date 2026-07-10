import asyncio
import re
from threading import RLock
from time import perf_counter

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.tools.base import BaseTool, ToolDefinition, ToolDescriptor
from backend.app.tools.providers.base import BaseToolProvider
from backend.app.tools.providers.builtin import BuiltinToolProvider
from backend.app.tools.providers.http import HTTPToolProvider
from backend.app.tools.providers.mcp import MCPToolProvider
from backend.app.tools.providers.plugin import PluginToolProvider


class ToolRegistryError(RuntimeError):
    pass


class ToolDuplicateError(ToolRegistryError):
    pass


class ToolDisabledError(ToolRegistryError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._descriptors: dict[str, ToolDescriptor] = {}
        self._providers: dict[str, BaseToolProvider] = {}
        self._lock = RLock()
        self._async_lock = asyncio.Lock()
        self._version = 0
        self._last_refresh: dict | None = None

    @property
    def version(self) -> int:
        return self._version

    @property
    def last_refresh(self) -> dict | None:
        return self._last_refresh

    def register(
        self,
        tool: BaseTool,
        descriptor: ToolDescriptor | None = None,
        *,
        replace: bool = False,
    ) -> None:
        self._validate_tool_name(tool.name)
        with self._lock:
            if tool.name in self._tools and not replace:
                raise ToolDuplicateError(f"tool already registered: {tool.name}")
            tool_descriptor = descriptor or tool.get_descriptor()
            if tool_descriptor.name != tool.name:
                tool_descriptor = tool_descriptor.model_copy(update={"name": tool.name})
            self._tools[tool.name] = tool
            self._descriptors[tool.name] = tool_descriptor
            self._bump_version()

    def unregister(self, tool_name: str) -> None:
        with self._lock:
            self._tools.pop(tool_name, None)
            self._descriptors.pop(tool_name, None)
            self._bump_version()

    def enable(self, tool_name: str) -> None:
        with self._lock:
            descriptor = self._descriptors.get(tool_name)
            if descriptor is None:
                raise KeyError(f"tool not found: {tool_name}")
            self._descriptors[tool_name] = descriptor.model_copy(update={"enabled": True})
            self._bump_version()

    def disable(self, tool_name: str) -> None:
        with self._lock:
            descriptor = self._descriptors.get(tool_name)
            if descriptor is None:
                raise KeyError(f"tool not found: {tool_name}")
            self._descriptors[tool_name] = descriptor.model_copy(update={"enabled": False})
            self._bump_version()

    def contains(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get_tool(self, name: str, *, require_enabled: bool = False) -> BaseTool | None:
        descriptor = self._descriptors.get(name)
        if require_enabled and descriptor is not None and not descriptor.enabled:
            raise ToolDisabledError(f"tool disabled: {name}")
        return self._tools.get(name)

    def get(self, name: str) -> BaseTool | None:
        return self.get_tool(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def list_enabled_tools(self) -> list[BaseTool]:
        return [
            tool
            for tool in self._tools.values()
            if self._descriptors.get(tool.name, tool.get_descriptor()).enabled
        ]

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [
            descriptor.to_definition()
            for descriptor in self.list_descriptors(enabled_only=True)
        ]

    def list_descriptors(self, *, enabled_only: bool = False) -> list[ToolDescriptor]:
        descriptors = list(self._descriptors.values())
        if enabled_only:
            descriptors = [descriptor for descriptor in descriptors if descriptor.enabled]
        return descriptors

    def get_descriptor(self, tool_name: str) -> ToolDescriptor | None:
        return self._descriptors.get(tool_name)

    def register_provider(self, provider: BaseToolProvider, *, replace: bool = True) -> None:
        with self._lock:
            if provider.name in self._providers and not replace:
                raise ToolDuplicateError(f"provider already registered: {provider.name}")
            self._providers[provider.name] = provider
            self._bump_version()

    def unregister_provider(self, provider_name: str) -> None:
        with self._lock:
            self._providers.pop(provider_name, None)
            self._bump_version()

    def list_providers(self) -> list[dict]:
        return [provider.health() for provider in self._providers.values()]

    def refresh(self) -> dict:
        started_at = perf_counter()
        result = {
            "discovered": 0,
            "added": 0,
            "updated": 0,
            "removed": 0,
            "failed": 0,
            "errors": [],
            "duration_ms": 0,
        }
        discovered_names: set[str] = set()
        provider_names = set(self._providers)
        failed_providers: set[str] = set()

        for provider in list(self._providers.values()):
            try:
                tools = provider.discover()
                result["discovered"] += len(tools)
                for tool in tools:
                    discovered_names.add(tool.name)
                    descriptor = tool.get_descriptor().model_copy(
                        update={"provider": provider.name}
                    )
                    exists = self.contains(tool.name)
                    self.register(tool, descriptor=descriptor, replace=True)
                    result["updated" if exists else "added"] += 1
            except Exception as exc:
                failed_providers.add(provider.name)
                result["failed"] += 1
                result["errors"].append({"provider": provider.name, "error": str(exc)})
                logger.exception(f"Tool provider discovery failed | provider={provider.name}")

        for tool_name, descriptor in list(self._descriptors.items()):
            if (
                descriptor.provider in provider_names
                and descriptor.provider not in failed_providers
                and tool_name not in discovered_names
            ):
                self.unregister(tool_name)
                result["removed"] += 1

        result["duration_ms"] = round((perf_counter() - started_at) * 1000, 2)
        self._last_refresh = result
        return result

    async def arefresh(self) -> dict:
        async with self._async_lock:
            return await asyncio.to_thread(self.refresh)

    def snapshot(self) -> dict:
        return {
            "registry_version": self.version,
            "last_refresh": self.last_refresh,
            "providers": self.list_providers(),
            "tools": [
                _sanitize_descriptor(descriptor).model_dump()
                for descriptor in self.list_descriptors()
            ],
        }

    def _bump_version(self) -> None:
        self._version += 1

    def _validate_tool_name(self, name: str) -> None:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name):
            raise ValueError(f"invalid tool name: {name}")


def _sanitize_descriptor(descriptor: ToolDescriptor) -> ToolDescriptor:
    metadata = {
        key: value
        for key, value in descriptor.metadata.items()
        if "key" not in key.lower()
        and "token" not in key.lower()
        and "secret" not in key.lower()
        and "password" not in key.lower()
        and "header" not in key.lower()
    }
    return descriptor.model_copy(update={"metadata": metadata})


_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _register_default_providers(_tool_registry)
        _tool_registry.refresh()
    return _tool_registry


def _register_default_providers(registry: ToolRegistry) -> None:
    registry.register_provider(BuiltinToolProvider())
    if settings.TOOL_PLUGIN_ENABLED:
        registry.register_provider(PluginToolProvider())
    if settings.HTTP_TOOL_PROVIDER_ENABLED:
        registry.register_provider(HTTPToolProvider())
    if settings.WORKFLOW_TOOL_PROVIDER_ENABLED:
        from backend.app.tools.providers.workflow import WorkflowToolProvider

        registry.register_provider(WorkflowToolProvider())
    if settings.MCP_TOOL_PROVIDER_ENABLED:
        registry.register_provider(MCPToolProvider())
