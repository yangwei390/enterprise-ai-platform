CUSTOMER_SERVICE_AGENT_ID = "customer_service_agent"
CUSTOMER_SERVICE_PLANNER_STRATEGY = "customer_service_rules"

CUSTOMER_SERVICE_TOOL_ALLOWLIST = [
    "search_products",
    "recommend_products",
    "compare_products",
    "knowledge_search",
    "query_order",
    "query_logistics",
    "create_after_sales_ticket",
    "create_human_handoff",
]

CUSTOMER_SERVICE_PENDING_KEY = "pending_after_sales"

CUSTOMER_SERVICE_PENDING_STATUS = "PENDING_CONFIRMATION"
CUSTOMER_SERVICE_CONFIRMING_STATUS = "CONFIRMING"
CUSTOMER_SERVICE_CONFIRMED_STATUS = "CONFIRMED"
CUSTOMER_SERVICE_CANCELLED_STATUS = "CANCELLED"
CUSTOMER_SERVICE_INVALIDATED_STATUS = "INVALIDATED"


def customer_service_allowed_knowledge_base_ids(
    *,
    agent_id: str | None,
    configured_ids: str,
) -> frozenset[int]:
    if agent_id != CUSTOMER_SERVICE_AGENT_ID:
        return frozenset()
    result: set[int] = set()
    for raw_value in configured_ids.split(","):
        value = raw_value.strip()
        if not value:
            continue
        knowledge_base_id = int(value)
        if knowledge_base_id < 1:
            raise ValueError("Customer service knowledge base IDs must be positive")
        result.add(knowledge_base_id)
    return frozenset(result)
