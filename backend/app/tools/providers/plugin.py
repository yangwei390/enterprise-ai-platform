import importlib.util
from pathlib import Path
from types import ModuleType

from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.logger import logger
from backend.app.tools.base import BaseTool
from backend.app.tools.providers.base import BaseToolProvider


class PluginToolProvider(BaseToolProvider):
    def __init__(self, plugin_path: str | None = None) -> None:
        self.plugin_path = Path(plugin_path or settings.TOOL_PLUGIN_PATH)
        if not self.plugin_path.is_absolute():
            self.plugin_path = PROJECT_ROOT / self.plugin_path
        self.errors: list[str] = []

    @property
    def name(self) -> str:
        return "plugin"

    def discover(self) -> list[BaseTool]:
        self.errors = []
        if not settings.TOOL_PLUGIN_ENABLED:
            return []
        if not self._is_safe_plugin_path() or not self.plugin_path.exists():
            return []

        tools: list[BaseTool] = []
        for file_path in sorted(self.plugin_path.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            try:
                tools.extend(self._load_module_tools(file_path))
            except Exception as exc:
                error = f"{file_path.name}: {exc}"
                self.errors.append(error)
                logger.exception(f"Plugin tool discovery failed | plugin={file_path}")
        return tools

    def health(self) -> dict:
        return {
            "provider": self.name,
            "healthy": not self.errors,
            "errors": self.errors,
            "plugin_path": str(self.plugin_path),
        }

    def _is_safe_plugin_path(self) -> bool:
        try:
            resolved_path = self.plugin_path.resolve()
            project_root = PROJECT_ROOT.resolve()
        except FileNotFoundError:
            return False
        return resolved_path == project_root or project_root in resolved_path.parents

    def _load_module_tools(self, file_path: Path) -> list[BaseTool]:
        module_name = f"backend.app.tools.plugins.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return self._extract_tools(module)

    def _extract_tools(self, module: ModuleType) -> list[BaseTool]:
        if hasattr(module, "create_tool"):
            tool = module.create_tool()
            return [tool] if isinstance(tool, BaseTool) else []

        raw_tools = getattr(module, "TOOLS", [])
        if not isinstance(raw_tools, list):
            return []
        return [tool for tool in raw_tools if isinstance(tool, BaseTool)]
