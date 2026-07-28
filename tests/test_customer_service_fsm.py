from backend.app.agents.customer_service_core.actions import (
    allowed_tools,
    is_tool_allowed,
)
from backend.app.agents.customer_service_core.dispatcher import (
    DispatchPhase,
    build_dispatch_plan,
)
from backend.app.agents.customer_service_core.dst import (
    load_dst,
    mutate_dst,
    replace_domain_candidates,
)
from backend.app.agents.customer_service_core.fsm import (
    FSMAction,
    apply_request,
    next_directive,
)
from backend.app.agents.customer_service_core.schemas import (
    AfterSalesPayload,
    ContextualizedRequest,
    ConversationDST,
    CustomerServiceDomain,
    CustomerServiceIntent,
    CustomerServiceSource,
    DialogStatus,
    GeneralPayload,
    HandoffPayload,
    KnowledgePayload,
    OrderAction,
    OrderPayload,
    OrderScope,
    PendingConfirmation,
    ProductPayload,
    TargetResolutionSource,
)


def _request(
    *,
    intent: CustomerServiceIntent,
    domain: CustomerServiceDomain,
    action: str,
    payload: (
        ProductPayload
        | OrderPayload
        | KnowledgePayload
        | AfterSalesPayload
        | HandoffPayload
        | GeneralPayload
    ),
) -> ContextualizedRequest:
    return ContextualizedRequest(
        raw_query="test",
        rewritten_query="test",
        domain=domain,
        intent=intent,
        action=action,
        source=CustomerServiceSource.PLANNER,
        payload=payload,
    )


def test_product_fact_requires_validated_target_and_attribute() -> None:
    request = _request(
        intent=CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        domain=CustomerServiceDomain.PRODUCT,
        action="query_product_document",
        payload=ProductPayload(
            target_product_codes=["p-1"],
            attributes=["buttons"],
            resolution_source=TargetResolutionSource.EXPLICIT,
        ),
    )
    dst = ConversationDST()

    apply_request(dst, request)

    assert dst.status == DialogStatus.READY_TO_EXECUTE
    directive = next_directive(dst, request)
    assert directive.action == FSMAction.EXECUTE
    plan = build_dispatch_plan(request, directive)
    assert plan.phase == DispatchPhase.EXECUTE
    assert plan.allowed_tools == frozenset({"search_products", "knowledge_search"})
    assert dst.slots["product_ref"].value == "p-1"


def test_logistics_without_target_collects_order_ref() -> None:
    request = _request(
        intent=CustomerServiceIntent.LOGISTICS_QUERY,
        domain=CustomerServiceDomain.LOGISTICS,
        action=OrderAction.LOGISTICS,
        payload=OrderPayload(
            action=OrderAction.LOGISTICS,
            scope=OrderScope.SELECTED,
        ),
    )
    dst = ConversationDST()

    apply_request(dst, request)
    directive = next_directive(dst, request)

    assert dst.status == DialogStatus.COLLECTING_SLOTS
    assert directive.action == FSMAction.COLLECT_SLOTS
    assert directive.missing_slots == ["order_ref"]


def test_after_sales_executes_draft_after_slots_are_complete() -> None:
    request = _request(
        intent=CustomerServiceIntent.AFTER_SALES,
        domain=CustomerServiceDomain.AFTER_SALES,
        action="after_sales",
        payload=AfterSalesPayload(
            order_ref="order-1",
            customer_phone_last4="1234",
            issue_type="refund",
            reason="damaged",
        ),
    )
    dst = ConversationDST()

    apply_request(dst, request)

    assert dst.status == DialogStatus.READY_TO_EXECUTE
    assert next_directive(dst, request).action == FSMAction.EXECUTE


def test_after_sales_pending_draft_waits_for_explicit_confirmation() -> None:
    request = _request(
        intent=CustomerServiceIntent.AFTER_SALES,
        domain=CustomerServiceDomain.AFTER_SALES,
        action="after_sales",
        payload=AfterSalesPayload(confirmed=True),
    )
    dst = ConversationDST(
        pending_confirmation=PendingConfirmation(
            operation_id="operation-1",
            action="after_sales",
            status="pending_confirmation",
        )
    )

    apply_request(dst, request)

    assert dst.status == DialogStatus.WAITING_CONFIRMATION
    directive = next_directive(dst, request)
    assert directive.action == FSMAction.WAIT_CONFIRMATION
    assert build_dispatch_plan(request, directive).phase == DispatchPhase.WAIT


def test_domain_interruption_pushes_active_task_to_stack() -> None:
    dst = ConversationDST(
        active_domain=CustomerServiceDomain.PRODUCT,
        active_intent=CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        status=DialogStatus.COLLECTING_SLOTS,
    )
    request = _request(
        intent=CustomerServiceIntent.POLICY_QUESTION,
        domain=CustomerServiceDomain.KNOWLEDGE,
        action="query_policy",
        payload=KnowledgePayload(question="退货规则"),
    )

    apply_request(dst, request)

    assert dst.active_domain == CustomerServiceDomain.KNOWLEDGE
    assert len(dst.stack) == 1
    assert dst.stack[0].domain == CustomerServiceDomain.PRODUCT
    assert dst.stack[0].status == DialogStatus.INTERRUPTED


def test_domain_candidates_are_isolated_and_persisted() -> None:
    metadata: dict[str, object] = {}

    def seed(dst: ConversationDST) -> None:
        replace_domain_candidates(
            dst,
            domain=CustomerServiceDomain.PRODUCT,
            candidates=[{"ref": "p-1", "display_name": "Product"}],
            active_ref="p-1",
        )
        replace_domain_candidates(
            dst,
            domain=CustomerServiceDomain.ORDER,
            candidates=[{"ref": "o-1", "display_name": "Order"}],
            active_ref="o-1",
        )

    mutate_dst(metadata, seed)
    restored = load_dst(metadata)

    assert restored.domains[CustomerServiceDomain.PRODUCT].active_ref == "p-1"
    assert restored.domains[CustomerServiceDomain.ORDER].active_ref == "o-1"


def test_handoff_requires_reason_but_not_concrete_catalog_rules() -> None:
    request = _request(
        intent=CustomerServiceIntent.HUMAN_HANDOFF,
        domain=CustomerServiceDomain.HUMAN_HANDOFF,
        action="human_handoff",
        payload=HandoffPayload(
            order_ref="order-1",
            customer_phone_last4="1234",
            reason="customer_request",
        ),
    )
    dst = ConversationDST()

    apply_request(dst, request)

    assert next_directive(dst, request).action == FSMAction.EXECUTE


def test_action_map_keeps_manual_lookup_inside_product_evidence_chain() -> None:
    tools = allowed_tools(CustomerServiceIntent.PRODUCT_DOCUMENT_FACT)

    assert tools == frozenset({"search_products", "knowledge_search"})
    assert not is_tool_allowed(
        CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        "query_order",
    )
