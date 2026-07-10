from backend.app.tools.base import BaseTool
from backend.app.tools.providers import BuiltinToolProvider


def load_builtin_tools() -> list[BaseTool]:
    return BuiltinToolProvider().discover()
