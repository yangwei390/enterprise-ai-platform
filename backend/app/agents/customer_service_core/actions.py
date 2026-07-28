from __future__ import annotations

from backend.app.agents.customer_service_core.schemas import CustomerServiceIntent

_INTENT_TOOLS: dict[CustomerServiceIntent, frozenset[str]] = {
    CustomerServiceIntent.GREETING: frozenset(),
    CustomerServiceIntent.PRODUCT_RECOMMENDATION: frozenset(
        {"recommend_products", "search_products"}
    ),
    CustomerServiceIntent.PRODUCT_SEARCH: frozenset({"search_products"}),
    CustomerServiceIntent.PRODUCT_REALTIME_FACT: frozenset({"search_products"}),
    CustomerServiceIntent.PRODUCT_DOCUMENT_FACT: frozenset({"search_products", "knowledge_search"}),
    CustomerServiceIntent.PRODUCT_COMPARISON: frozenset({"compare_products"}),
    CustomerServiceIntent.POLICY_QUESTION: frozenset({"knowledge_search"}),
    CustomerServiceIntent.ORDER_QUERY: frozenset({"query_order"}),
    CustomerServiceIntent.LOGISTICS_QUERY: frozenset({"query_order", "query_logistics"}),
    CustomerServiceIntent.AFTER_SALES: frozenset({"create_after_sales_ticket"}),
    CustomerServiceIntent.HUMAN_HANDOFF: frozenset({"create_human_handoff"}),
    CustomerServiceIntent.OUT_OF_SCOPE: frozenset(),
    CustomerServiceIntent.OTHER: frozenset(),
}


def allowed_tools(intent: CustomerServiceIntent) -> frozenset[str]:
    return _INTENT_TOOLS[intent]


def is_tool_allowed(intent: CustomerServiceIntent, tool_name: str) -> bool:
    return tool_name in allowed_tools(intent)
