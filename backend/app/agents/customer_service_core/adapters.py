from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    AfterSalesCommand,
    CompareProductsCommand,
    CustomerServiceCommand,
    HumanHandoffCommand,
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
            arguments["page_size"] = command.page_size
            return command.kind, arguments
        if isinstance(command, RecommendProductsCommand):
            return command.kind, {**command.filters, "page_size": command.page_size}
        if isinstance(command, CompareProductsCommand):
            arguments: dict[str, Any] = {"product_codes": command.product_codes}
            if command.fields:
                arguments["fields"] = command.fields
            return command.kind, arguments
        if isinstance(command, KnowledgeSearchCommand):
            return command.kind, command.model_dump(exclude={"kind"}, exclude_none=True)
        if isinstance(command, QueryOrderCommand):
            return command.kind, command.model_dump(exclude={"kind"}, exclude_none=True)
        if isinstance(command, QueryLogisticsCommand):
            return command.kind, command.model_dump(exclude={"kind"})
        if isinstance(command, (AfterSalesCommand, HumanHandoffCommand)):
            return command.kind, dict(command.arguments)
        raise TypeError(f"unsupported customer service command: {type(command).__name__}")


def command_from_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> CustomerServiceCommand:
    if tool_name == "search_products":
        known = {"keyword", "category", "product_code", "page_size"}
        return SearchProductsCommand(
            keyword=arguments.get("keyword"),
            category=arguments.get("category"),
            product_code=arguments.get("product_code"),
            page_size=arguments.get("page_size", 20),
            filters={key: value for key, value in arguments.items() if key not in known},
        )
    if tool_name == "recommend_products":
        return RecommendProductsCommand(
            page_size=arguments.get("page_size", 3),
            filters={
                key: value
                for key, value in arguments.items()
                if key != "page_size"
            },
        )
    if tool_name == "compare_products":
        return CompareProductsCommand(
            product_codes=arguments.get("product_codes", []),
            fields=arguments.get("fields", []),
        )
    if tool_name == "knowledge_search":
        return KnowledgeSearchCommand.model_validate({"kind": tool_name, **arguments})
    if tool_name == "query_order":
        return QueryOrderCommand(order_ref=arguments.get("order_ref"))
    if tool_name == "query_logistics":
        return QueryLogisticsCommand.model_validate({"kind": tool_name, **arguments})
    if tool_name == "create_after_sales_ticket":
        return AfterSalesCommand(arguments=arguments)
    if tool_name == "create_human_handoff":
        return HumanHandoffCommand(arguments=arguments)
    raise ValueError(f"unsupported customer service tool: {tool_name}")
