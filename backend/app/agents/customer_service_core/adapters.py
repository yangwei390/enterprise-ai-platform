from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CompareProductsCommand,
    ConfirmAfterSalesCommand,
    CreateAfterSalesDraftCommand,
    CreateHumanHandoffCommand,
    CustomerServiceCommand,
    KnowledgeSearchCommand,
    QueryLogisticsCommand,
    QueryOrderCommand,
    RecommendProductsCommand,
    SearchProductsCommand,
)
from backend.app.tools.registry import ToolRegistry, get_tool_registry


class CommandAdapter:
    """Command 到真实 Tool 参数的唯一入口。"""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def adapt(self, command: CustomerServiceCommand) -> tuple[str, dict[str, Any]]:
        tool_name, raw_arguments = self._raw_arguments(command)
        tool = self.registry.get_tool(tool_name, require_enabled=True)
        if tool is None:
            raise ValueError(f"tool not found: {tool_name}")
        validated = tool.args_schema.model_validate(raw_arguments)
        return tool_name, validated.model_dump(exclude_none=True)

    def _raw_arguments(
        self,
        command: CustomerServiceCommand,
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(command, SearchProductsCommand):
            arguments = {**command.filters}
            if command.keyword is not None:
                arguments["keyword"] = command.keyword
            if command.category is not None:
                arguments["category"] = command.category
            if command.product_code is not None:
                arguments["product_code"] = command.product_code
            if command.knowledge_base_id is not None:
                arguments["knowledge_base_id"] = command.knowledge_base_id
            arguments["page_size"] = command.page_size
            return command.kind, arguments
        if isinstance(command, RecommendProductsCommand):
            arguments = {**command.filters, "page_size": command.page_size}
            if command.knowledge_base_id is not None:
                arguments["knowledge_base_id"] = command.knowledge_base_id
            return command.kind, arguments
        if isinstance(command, CompareProductsCommand):
            arguments: dict[str, Any] = {"product_codes": command.product_codes}
            if command.fields:
                arguments["fields"] = command.fields
            if command.knowledge_base_id is not None:
                arguments["knowledge_base_id"] = command.knowledge_base_id
            return command.kind, arguments
        if isinstance(command, KnowledgeSearchCommand):
            return command.kind, command.model_dump(exclude={"kind"}, exclude_none=True)
        if isinstance(command, QueryOrderCommand):
            return command.kind, command.model_dump(exclude={"kind"}, exclude_none=True)
        if isinstance(command, QueryLogisticsCommand):
            return command.kind, command.model_dump(exclude={"kind"})
        if isinstance(
            command,
            (CreateAfterSalesDraftCommand, ConfirmAfterSalesCommand),
        ):
            return (
                "create_after_sales_ticket",
                command.model_dump(exclude={"kind"}, exclude_none=True),
            )
        if isinstance(command, CreateHumanHandoffCommand):
            return (
                "create_human_handoff",
                command.model_dump(exclude={"kind"}, exclude_none=True),
            )
        raise TypeError(f"unsupported customer service command: {type(command).__name__}")
