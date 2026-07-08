from backend.app.tools.base import BaseTool
from backend.app.tools.builtin import (
    CalculatorTool,
    CurrentTimeTool,
    EchoTool,
    KnowledgeSearchTool,
)


def load_builtin_tools() -> list[BaseTool]:
    return [CalculatorTool(), EchoTool(), CurrentTimeTool(), KnowledgeSearchTool()]
