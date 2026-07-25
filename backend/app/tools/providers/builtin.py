from backend.app.tools.base import BaseTool
from backend.app.tools.builtin import (
    CalculatorTool,
    CompareProductsTool,
    CreateAfterSalesTicketTool,
    CreateHumanHandoffTool,
    CurrentTimeTool,
    EchoTool,
    KnowledgeSearchTool,
    QueryLogisticsTool,
    QueryOrderTool,
    RecommendProductsTool,
    SearchProductsTool,
)
from backend.app.tools.providers.base import BaseToolProvider


class BuiltinToolProvider(BaseToolProvider):
    @property
    def name(self) -> str:
        return "builtin"

    def discover(self) -> list[BaseTool]:
        return [
            CalculatorTool(),
            EchoTool(),
            CurrentTimeTool(),
            KnowledgeSearchTool(),
            SearchProductsTool(),
            RecommendProductsTool(),
            CompareProductsTool(),
            QueryOrderTool(),
            QueryLogisticsTool(),
            CreateAfterSalesTicketTool(),
            CreateHumanHandoffTool(),
        ]
