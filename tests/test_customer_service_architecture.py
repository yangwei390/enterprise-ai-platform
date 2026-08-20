from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from backend.app.agents.customer_service_contract import (
    CUSTOMER_SERVICE_AGENT_ID,
    CUSTOMER_SERVICE_PENDING_KEY,
)
from backend.app.agents.customer_service_core.adapters import (
    CommandAdapter,
)
from backend.app.agents.customer_service_core.after_sales_guard import (
    confirmation_coordinator,
)
from backend.app.agents.customer_service_core.capability_selector import (
    CapabilitySuggestion,
)
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.agents.customer_service_core.context_lifecycle import (
    reconcile_visible_context,
)
from backend.app.agents.customer_service_core.contracts import (
    CandidateProduct,
    CompareProductsCommand,
    ConfirmAfterSalesCommand,
    CreateAfterSalesDraftCommand,
    CreateHumanHandoffCommand,
    CustomerServiceExecution,
    CustomerServiceState,
    DialogFocus,
    ExecutionPhase,
    GoalSnapshot,
    KnowledgeSearchCommand,
    OrderCandidate,
    OrderCandidateBatch,
    OrderContext,
    PendingAfterSales,
    PendingClarification,
    PendingProductQuery,
    PendingTransaction,
    ProductCandidateBatch,
    ProductContext,
    ProductQuestionFocus,
    QueryLogisticsCommand,
    QueryOrderCommand,
    RecommendProductsCommand,
    ReferenceExpression,
    SearchProductsCommand,
)
from backend.app.agents.customer_service_core.entity_resolver import (
    ResolutionStatus,
    resolve_product_reference,
)
from backend.app.agents.customer_service_core.hooks import evaluate_tool_policy
from backend.app.agents.customer_service_core.presenter import CustomerServicePresenter
from backend.app.agents.customer_service_core.reducer import (
    preview_product_filters,
    reduce_state,
)
from backend.app.agents.customer_service_core.strategy import CustomerServiceStrategy
from backend.app.agents.customer_service_core.understanding import understand
from backend.app.agents.langgraph.budget import AgentExecutionBudget
from backend.app.agents.langgraph.nodes import FinalNode, ObservationNode, PlannerNode, ToolNode
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.config import settings
from backend.app.memory.state import MemoryState
from backend.app.tools.base import ToolResult
from backend.app.tools.builtin.customer_service import (
    CompareProductsTool,
    CreateAfterSalesTicketTool,
    CreateHumanHandoffTool,
    QueryLogisticsTool,
    QueryOrderTool,
    RecommendProductsTool,
    SearchProductsTool,
)
from backend.app.tools.builtin.knowledge_tool import KnowledgeSearchTool
from backend.app.tools.registry import ToolRegistry

_FORMAL_STAGE_SETTINGS = (
    "CUSTOMER_SERVICE_QUERY_REWRITE_MODE",
    "CUSTOMER_SERVICE_INTENT_ROUTING_MODE",
    "CUSTOMER_SERVICE_SLOT_EXTRACTION_MODE",
    "CUSTOMER_SERVICE_TOOL_SELECTION_MODE",
    "CUSTOMER_SERVICE_REFERENCE_INTERPRETATION_MODE",
)


def _set_formal_stage_modes(monkeypatch, mode: str) -> None:
    for setting_name in _FORMAL_STAGE_SETTINGS:
        monkeypatch.setattr(settings, setting_name, mode)


@pytest.fixture(autouse=True)
def _default_formal_stage_modes_to_rules(monkeypatch):
    _set_formal_stage_modes(monkeypatch, "rule_only")
    confirmation_coordinator.reset()


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        SearchProductsTool(),
        RecommendProductsTool(),
        CompareProductsTool(),
        KnowledgeSearchTool(),
        QueryOrderTool(),
        QueryLogisticsTool(),
        CreateAfterSalesTicketTool(),
        CreateHumanHandoffTool(),
    ):
        registry.register(tool)
    return registry


@pytest.mark.parametrize(
    "command",
    [
        SearchProductsCommand(product_code="P-1"),
        RecommendProductsCommand(filters={"category": "键盘"}, page_size=2),
        CompareProductsCommand(product_codes=["P-1", "P-2"]),
        KnowledgeSearchCommand(
            query="如何连接",
            knowledge_base_id=1,
            document_id=2,
        ),
        QueryOrderCommand(),
        QueryLogisticsCommand(order_ref="ORDER-001"),
        CreateAfterSalesDraftCommand(
            order_no="ORDER-001",
            customer_phone_last4="1234",
            issue_type="return",
            issue_description="商品无法正常使用",
        ),
        ConfirmAfterSalesCommand(
            order_no="ORDER-001",
            customer_phone_last4="1234",
            draft_id="mock-draft-0123456789abcdef01234567",
            operation_id="mock-draft-0123456789abcdef01234567",
        ),
        CreateHumanHandoffCommand(
            order_no="ORDER-001",
            customer_phone_last4="1234",
            reason="customer_request",
            message="需要人工协助",
        ),
    ],
)
def test_every_command_validates_against_real_tool_schema(command) -> None:
    tool_name, arguments = CommandAdapter(_registry()).adapt(command)
    tool = _registry().get_tool(tool_name, require_enabled=True)
    assert tool is not None
    tool.args_schema.model_validate(arguments)


def test_category_switch_removes_all_old_filters() -> None:
    state = CustomerServiceState(
        filters={
            "category": "鼠标和指针设备",
            "brand": "Logitech",
            "price_max": 500,
            "model": "M1",
            "required_features": ["静音"],
            "required_use_cases": ["办公"],
        }
    )
    preview, patch = preview_product_filters(
        state,
        {"category": "键盘", "required_features": ["机械轴"], "keyword": "无线"},
    )
    assert state.product.filters["category"] == "鼠标和指针设备"
    assert preview.product.filters == {
        "category": "键盘",
        "required_features": ["机械轴"],
        "keyword": "无线",
    }
    assert patch == {
        "product_filters": preview.product.filters,
        "invalidate_product_context": True,
    }


def test_product_context_keeps_only_one_active_batch() -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-1",
        query="推荐鼠标",
        category="鼠标和指针设备",
        items=[
            CandidateProduct(
                product_code="M-1",
                name="Mouse One",
                category="鼠标和指针设备",
                batch_id="mouse-1",
                position=0,
            )
        ],
    )
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            active_batch=batch,
            active_product_code="M-1",
        )
    )

    assert state.product.active_batch == batch
    assert not hasattr(state.product, "candidate_batches")


def test_customer_service_state_uses_typed_domain_contexts() -> None:
    state = CustomerServiceState()

    assert state.order == OrderContext()
    assert state.dialog_focus == DialogFocus()
    assert state.pending_clarification is None
    dumped = state.model_dump(mode="json")
    assert "order_candidates" not in dumped
    assert "active_order_ref" not in dumped
    assert "clarification_target" not in dumped
    assert "clarification_rounds" not in dumped


def test_legacy_order_state_migrates_to_typed_order_context() -> None:
    state = CustomerServiceState.model_validate(
        {
            "order_candidates": [
                {"order_no": "ORDER-001", "status": "delivered"},
                {"order_ref": "ORDER-002", "status": "cancelled"},
            ],
            "active_order_ref": "ORDER-002",
        }
    )

    assert state.order.active_order_ref == "ORDER-002"
    assert state.order.active_batch is not None
    assert [item.order_ref for item in state.order.active_batch.items] == [
        "ORDER-001",
        "ORDER-002",
    ]
    assert state.order.active_batch.items[0].details == {
        "order_no": "ORDER-001",
        "status": "delivered",
    }
    assert "order_candidates" not in state.model_dump(mode="json")


def test_legacy_clarification_state_migrates_to_typed_pending_contract() -> None:
    state = CustomerServiceState.model_validate(
        {
            "clarification_target": "product.category",
            "clarification_rounds": 2,
        }
    )

    assert state.pending_clarification == PendingClarification(
        clarification_id="legacy-clarification",
        kind="missing_slot",
        domain="general",
        target_description="product.category",
        missing_fields=["product.category"],
        attempts=2,
    )


def test_order_candidate_requires_real_reference() -> None:
    with pytest.raises(ValueError):
        OrderCandidate.model_validate(
            {
                "display_label": "订单",
                "batch_id": "orders-1",
                "position": 0,
            }
        )


def test_typed_order_context_round_trip() -> None:
    state = CustomerServiceState(
        order=OrderContext(
            active_batch=OrderCandidateBatch(
                batch_id="orders-1",
                query="我的订单",
                items=[
                    OrderCandidate(
                        order_ref="ORDER-001",
                        display_label="ORDER-001",
                        batch_id="orders-1",
                        position=0,
                        details={"status": "delivered"},
                    )
                ],
            ),
            active_order_ref="ORDER-001",
        )
    )

    restored = CustomerServiceState.model_validate(state.model_dump(mode="json"))
    assert restored == state


def test_candidate_batches_record_source_turn_without_fixed_ttl() -> None:
    product_batch = ProductCandidateBatch(
        batch_id="products-1",
        query="推荐鼠标",
        source_turn_id="turn-product",
    )
    order_batch = OrderCandidateBatch(
        batch_id="orders-1",
        query="我的订单",
        source_turn_id="turn-order",
    )

    assert product_batch.source_turn_id == "turn-product"
    assert order_batch.source_turn_id == "turn-order"
    assert not hasattr(product_batch, "expires_after_turn_sequence")
    assert not hasattr(order_batch, "expires_after_turn_sequence")


def test_category_switch_preview_clears_old_product_context() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            filters={"category": "鼠标和指针设备", "price_max": 300},
            active_batch=ProductCandidateBatch(
                batch_id="mouse-1",
                query="推荐鼠标",
                category="鼠标和指针设备",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        category="鼠标和指针设备",
                        batch_id="mouse-1",
                        position=0,
                    )
                ],
            ),
            active_product_code="M-1",
            last_question=ProductQuestionFocus(
                predicate="bluetooth_connectivity",
                batch_id="mouse-1",
            ),
        )
    )

    preview, patch = preview_product_filters(
        state,
        {"category": "键盘", "keyword": "键盘"},
    )

    assert preview.product.active_category == "键盘"
    assert preview.product.active_batch is None
    assert preview.product.active_product_code is None
    assert preview.product.last_question is None
    assert preview.product.filters == {"category": "键盘", "keyword": "键盘"}
    assert patch["invalidate_product_context"] is True


def test_same_category_recommendation_inherits_filters() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            filters={"category": "鼠标和指针设备", "price_max": 300},
        )
    )

    preview, patch = preview_product_filters(state, {})

    assert preview.product.filters == {
        "category": "鼠标和指针设备",
        "price_max": 300,
    }
    assert patch["invalidate_product_context"] is False


def test_continuation_excludes_active_batch_with_tool_subcategory() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            filters={
                "category": "鼠标和指针设备",
                "excluded_product_codes": ["1"],
            },
            active_batch=ProductCandidateBatch(
                batch_id="mouse-2",
                query="再推荐一个",
                category="鼠标和指针设备",
                items=[
                    CandidateProduct(
                        product_code="2",
                        name="MX Master 4",
                        category="办公鼠标",
                        batch_id="mouse-2",
                        position=0,
                    )
                ],
            ),
            active_product_code="2",
        )
    )

    result = asyncio.run(understand(query="还有其他的么", state=state, messages=[]))
    preview = reduce_state(state, result.frame)

    assert result.frame.continuation is True
    assert preview.proposed_state.product.filters["excluded_product_codes"] == [
        "1",
        "2",
    ]


@pytest.mark.parametrize("query", ["还有么", "其他的还有么", "别的还有吗"])
def test_common_alternative_phrasings_are_rule_continuations(query: str) -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="键盘",
            filters={"category": "键盘"},
        )
    )

    result = asyncio.run(understand(query=query, state=state, messages=[]))

    assert result.llm_used is False
    assert result.frame.intent == "recommend_products"
    assert result.frame.continuation is True


def test_ordinal_cannot_resolve_a_different_category() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="键盘",
            active_batch=ProductCandidateBatch(
                batch_id="keyboard-1",
                query="推荐键盘",
                category="键盘",
                items=[
                    CandidateProduct(
                        product_code="K-1",
                        name="Keyboard One",
                        category="键盘",
                        batch_id="keyboard-1",
                        position=0,
                    )
                ],
            ),
        )
    )

    resolution = resolve_product_reference(
        ReferenceExpression(text="第一个鼠标", ordinal=0, category_hint="鼠标"),
        state,
    )

    assert resolution.status == ResolutionStatus.NOT_FOUND


def test_confirmation_uses_pending_product_query_without_llm() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            pending_query=PendingProductQuery(
                category="鼠标和指针设备",
                keyword="鼠标",
                requested_count=1,
                created_turn_id="turn-1",
            )
        )
    )

    result = asyncio.run(understand(query="对", state=state, messages=[]))

    assert result.llm_used is False
    assert result.frame.intent == "confirm_pending_product_query"
    assert result.frame.slots == {"confirmation": True}


def test_ordinal_only_followup_inherits_question_inside_current_batch() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            active_batch=ProductCandidateBatch(
                batch_id="mouse-1",
                query="推荐鼠标",
                category="鼠标和指针设备",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        category="鼠标和指针设备",
                        batch_id="mouse-1",
                        position=0,
                    ),
                    CandidateProduct(
                        product_code="M-2",
                        name="Mouse Two",
                        category="鼠标和指针设备",
                        batch_id="mouse-1",
                        position=1,
                    ),
                ],
            ),
            last_question=ProductQuestionFocus(
                predicate="bluetooth_connectivity",
                batch_id="mouse-1",
                source_turn_id="turn-previous",
                source_question="第二个支持蓝牙吗",
            ),
        )
    )
    messages = [
        {"role": "user", "content": "第二个支持蓝牙吗"},
        {"role": "assistant", "content": "已根据说明书回答。"},
        {"role": "user", "content": "第一个呢"},
    ]

    result = asyncio.run(understand(query="第一个呢", state=state, messages=messages))

    assert result.llm_used is False
    assert result.frame.intent == "product_fact"
    assert result.frame.question == "该商品支持蓝牙吗"
    assert result.frame.references[0].ordinal == 0


def test_ordinal_only_after_product_list_opens_catalog_detail_without_rag() -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-list",
        query="你家卖鼠标么",
        category="鼠标和指针设备",
        items=[
            CandidateProduct(
                product_code="M-1",
                name="Mouse One",
                category="鼠标和指针设备",
                batch_id="mouse-list",
                position=0,
                primary_manual_document_id=9,
            ),
            CandidateProduct(
                product_code="M-2",
                name="Mouse Two",
                category="鼠标和指针设备",
                batch_id="mouse-list",
                position=1,
                primary_manual_document_id=10,
            ),
        ],
    )
    state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            filters={"keyword": "鼠标"},
            active_batch=batch,
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": state.model_dump(mode="json")},
    }

    planned = _plan_turn("第二个呢？", metadata)

    assert planned["state"]["customer_service_execution"]["goal"]["intent"] == (
        "product_catalog_detail"
    )
    assert planned["decision"].tool_calls[0].tool_name == "search_products"
    assert planned["decision"].tool_calls[0].arguments["product_code"] == "M-2"
    assert planned["state"]["customer_service_execution"]["pending_transaction"][
        "expected_result_type"
    ] == "product_catalog_detail"
    _commit_planned_products(
        planned["state"],
        planned["decision"],
        [
            {
                "product_code": "M-2",
                "name": "Mouse Two",
                "category": "鼠标和指针设备",
                "primary_manual_document_id": 10,
            }
        ],
    )
    committed = CustomerServiceState.model_validate(
        planned["state"]["metadata"]["customer_service"]["state"]
    )
    assert committed.product.active_batch == batch
    assert committed.product.active_product_code == "M-2"
    assert committed.product.last_question is None


def test_stale_question_focus_does_not_force_ordinal_into_rag() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            active_batch=ProductCandidateBatch(
                batch_id="mouse-list",
                query="你家卖鼠标么",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        batch_id="mouse-list",
                        position=0,
                    ),
                    CandidateProduct(
                        product_code="M-2",
                        name="Mouse Two",
                        batch_id="mouse-list",
                        position=1,
                    ),
                ],
            ),
            last_question=ProductQuestionFocus(
                predicate="bluetooth_connectivity",
                batch_id="mouse-list",
                source_turn_id="old-turn",
                source_question="之前的商品支持蓝牙吗",
            ),
        )
    )
    messages = [
        {"role": "user", "content": "你家卖鼠标么"},
        {"role": "assistant", "content": "1. Mouse One\n2. Mouse Two"},
        {"role": "user", "content": "第二个呢？"},
    ]

    result = asyncio.run(understand(query="第二个呢？", state=state, messages=messages))

    assert result.llm_used is False
    assert result.frame.intent == "product_catalog_detail"
    assert result.frame.requires_manual_evidence is False


def test_product_ordinal_expires_when_source_turn_leaves_message_window() -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-list",
        query="推荐两个鼠标",
        source_turn_id="turn-products",
        items=[
            CandidateProduct(
                product_code="M-1",
                name="Mouse One",
                batch_id="mouse-list",
                position=0,
            ),
            CandidateProduct(
                product_code="M-2",
                name="Mouse Two",
                batch_id="mouse-list",
                position=1,
            ),
        ],
    )
    business_state = CustomerServiceState(
        product=ProductContext(active_batch=batch, active_product_code="M-1"),
        dialog_focus=DialogFocus(
            active_domain="product",
            active_batch_id=batch.batch_id,
            source_turn_id="turn-products",
        ),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "runtime_turn_id": "turn-current",
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }
    state = {
        "query": "第二个呢",
        "messages": [
            {"role": "user", "content": "聊点别的", "turn_id": "turn-other"},
            {"role": "assistant", "content": "好的", "turn_id": "turn-other"},
            {"role": "user", "content": "第二个呢", "turn_id": "turn-current"},
        ],
        "metadata": metadata,
        "conversation_id": 52,
        "knowledge_base_id": 1,
        "allowed_knowledge_base_ids": [1],
    }

    decision = asyncio.run(CustomerServiceStrategy().adecide(state))

    assert decision.action == "final"
    committed = CustomerServiceState.model_validate(
        state["metadata"]["customer_service"]["state"]
    )
    assert committed.product.active_batch is None
    assert committed.product.active_product_code is None
    assert committed.dialog_focus.active_batch_id is None


def test_product_ordinal_remains_valid_while_source_turn_is_visible() -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-list",
        query="推荐两个鼠标",
        source_turn_id="turn-products",
        items=[
            CandidateProduct(
                product_code="M-1",
                name="Mouse One",
                batch_id="mouse-list",
                position=0,
            ),
            CandidateProduct(
                product_code="M-2",
                name="Mouse Two",
                batch_id="mouse-list",
                position=1,
            ),
        ],
    )
    business_state = CustomerServiceState(product=ProductContext(active_batch=batch))
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "runtime_turn_id": "turn-current",
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }
    state = {
        "query": "第二个呢",
        "messages": [
            {"role": "user", "content": "推荐两个鼠标", "turn_id": "turn-products"},
            {
                "role": "assistant",
                "content": "1. Mouse One\n2. Mouse Two",
                "turn_id": "turn-products",
            },
            {"role": "user", "content": "第二个呢", "turn_id": "turn-current"},
        ],
        "metadata": metadata,
        "conversation_id": 52,
        "knowledge_base_id": 1,
        "allowed_knowledge_base_ids": [1],
    }

    decision = asyncio.run(CustomerServiceStrategy().adecide(state))

    assert decision.tool_calls[0].arguments["product_code"] == "M-2"


def test_order_batch_expires_when_source_turn_leaves_message_window() -> None:
    batch = OrderCandidateBatch(
        batch_id="orders-1",
        query="我的订单",
        source_turn_id="turn-orders",
        items=[
            OrderCandidate(
                order_ref="ORDER-001",
                display_label="ORDER-001",
                batch_id="orders-1",
                position=0,
            )
        ],
    )
    state = CustomerServiceState(
        order=OrderContext(active_batch=batch, active_order_ref="ORDER-001"),
        dialog_focus=DialogFocus(
            active_domain="order",
            active_batch_id="orders-1",
            source_turn_id="turn-orders",
        ),
    )

    reconciled, changes = reconcile_visible_context(
        state,
        [{"role": "user", "content": "当前消息", "turn_id": "turn-current"}],
    )

    assert reconciled.order.active_batch is None
    assert reconciled.order.active_order_ref is None
    assert reconciled.dialog_focus.active_batch_id is None
    assert changes[0]["domain"] == "order"


def test_ordinal_does_not_search_an_old_product_batch() -> None:
    state = CustomerServiceState(
        candidate_batches=[
            ProductCandidateBatch(
                batch_id="mouse",
                query="推荐鼠标",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        category="鼠标和指针设备",
                        batch_id="mouse",
                        position=0,
                    )
                ],
            ),
            ProductCandidateBatch(
                batch_id="keyboard",
                query="推荐键盘",
                items=[
                    CandidateProduct(
                        product_code="K-1",
                        name="Keyboard One",
                        category="键盘",
                        batch_id="keyboard",
                        position=0,
                    )
                ],
            ),
        ]
    )
    resolution = resolve_product_reference(
        ReferenceExpression(text="第一款鼠标", ordinal=0, category_hint="鼠标"),
        state,
    )
    assert resolution.status == ResolutionStatus.NOT_FOUND


def test_explicit_entity_outside_pool_requires_tool_verification() -> None:
    resolution = resolve_product_reference(
        ReferenceExpression(text="P-404", explicit_code="P-404"),
        CustomerServiceState(),
    )
    assert resolution.status == ResolutionStatus.VERIFICATION_REQUIRED
    assert resolution.verification_query == {"product_code": "P-404"}


def test_compare_explicit_codes_builds_typed_compare_command() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }
    planned = _plan_turn("对比 P001 和 P002", metadata)
    call = planned["decision"].tool_calls[0]
    assert call.tool_name == "compare_products"
    assert call.arguments["product_codes"] == ["P001", "P002"]


def test_compare_two_ordinals_resolves_inside_same_candidate_batch() -> None:
    state = CustomerServiceState(
        candidate_batches=[
            ProductCandidateBatch(
                batch_id="mouse",
                query="推荐鼠标",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        category="鼠标和指针设备",
                        batch_id="mouse",
                        position=0,
                    ),
                    CandidateProduct(
                        product_code="M-2",
                        name="Mouse Two",
                        category="鼠标和指针设备",
                        batch_id="mouse",
                        position=1,
                    ),
                ],
            )
        ]
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": state.model_dump(mode="json")},
    }
    planned = _plan_turn("对比第一个和第二个", metadata)
    call = planned["decision"].tool_calls[0]
    assert call.tool_name == "compare_products"
    assert call.arguments["product_codes"] == ["M-1", "M-2"]


def test_failed_tool_never_pollutes_business_state() -> None:
    state = _agent_state_for(SearchProductsCommand(product_code="P-404"))
    before = deepcopy(state["metadata"]["customer_service"]["state"])
    committed = CommitCoordinator().commit(
        agent_state=state,
        tool_name="search_products",
        arguments={"product_code": "P-404"},
        result=ToolResult(name="search_products", success=False, error="not found"),
    )
    assert committed is False
    assert state["metadata"]["customer_service"]["state"] == before
    assert state["customer_service_execution"]["phase"] == "FAILED"


def test_result_contract_failure_never_pollutes_business_state() -> None:
    state = _agent_state_for(SearchProductsCommand(product_code="P-1"))
    before = deepcopy(state["metadata"]["customer_service"]["state"])
    committed = CommitCoordinator().commit(
        agent_state=state,
        tool_name="search_products",
        arguments={"product_code": "P-1"},
        result=ToolResult(
            name="search_products",
            success=True,
            result={"items": [{"name": "missing product code"}]},
        ),
    )
    assert committed is False
    assert state["metadata"]["customer_service"]["state"] == before


def test_verified_product_then_bound_manual_rag_end_to_end() -> None:
    command = SearchProductsCommand(product_code="P-1")
    state = _agent_state_for(command)
    execution = CustomerServiceExecution.model_validate(state["customer_service_execution"])
    execution.phase = ExecutionPhase.WAITING_TOOL
    execution.goal = GoalSnapshot(raw_query="P-1 支持蓝牙吗")
    execution.pending_transaction.expected_result_type = "product_verification_for_manual"
    state.update(
        {
            "messages": [{"role": "user", "content": "P-1 支持蓝牙吗"}],
            "tool_calls": [],
            "pending_tool_calls": [
                {
                    "id": "call-1",
                    "tool_name": "search_products",
                    "arguments": {"product_code": "P-1"},
                    "index": 0,
                }
            ],
            "tool_results": [],
            "observations": [],
            "step_count": 0,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "reflection_count": 0,
            "same_tool_repeat_count": 0,
            "loop_status": "running",
            "budget": AgentExecutionBudget.from_settings().model_dump(),
            "knowledge_base_id": 9,
            "allowed_knowledge_base_ids": [9],
            "customer_service_execution": execution.model_dump(mode="json"),
        }
    )

    asyncio.run(ToolNode(_ProductVerificationExecutor()).acall(state))
    asyncio.run(ObservationNode().acall(state))
    assert state["customer_service_execution"]["phase"] == "CONTINUE"

    decision = asyncio.run(CustomerServiceStrategy().adecide(state))
    assert decision.tool_calls[0].tool_name == "knowledge_search"
    assert decision.tool_calls[0].arguments["document_id"] == 77
    assert decision.tool_calls[0].arguments["knowledge_base_id"] == 9

    state["pending_tool_calls"] = [item.model_dump(mode="json") for item in decision.tool_calls]
    asyncio.run(ToolNode(_KnowledgeExecutor()).acall(state))
    asyncio.run(ObservationNode().acall(state))
    asyncio.run(FinalNode().acall(state))
    assert state["customer_service_execution"]["phase"] == "READY_FOR_FINAL"
    assert state["final_answer"] == "P-1 支持蓝牙连接。"
    assert state["tool_call_count"] == 2


def test_strategy_does_not_copy_stream_runtime_objects() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    future = asyncio.get_event_loop_policy().new_event_loop().create_future()
    state = {
        "query": "你好",
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {
            "_agent_stream_event_queue": queue,
            "_agent_stream_future": future,
            "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
        },
    }

    decision = asyncio.run(CustomerServiceStrategy().adecide(state))

    assert decision.content
    assert state["metadata"]["_agent_stream_event_queue"] is queue
    assert state["metadata"]["_agent_stream_future"] is future
    future.get_loop().close()


def test_product_multiturn_keeps_category_excludes_seen_and_resolves_latest() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    first = _plan_turn("给我推荐一个鼠标", metadata)
    assert first["decision"].tool_calls[0].arguments["category"] == "鼠标和指针设备"
    _commit_planned_products(
        first["state"],
        first["decision"],
        [
            {
                "product_code": "1",
                "name": "罗技G304",
                "category": "鼠标",
                "primary_manual_document_id": 9,
            }
        ],
    )

    second = _plan_turn("还有其他的么", metadata)
    second_args = second["decision"].tool_calls[0].arguments
    assert second_args["category"] == "鼠标和指针设备"
    assert second_args["excluded_product_codes"] == ["1"]
    _commit_planned_products(
        second["state"],
        second["decision"],
        [
            {
                "product_code": "2",
                "name": "MX Master 4",
                "category": "办公鼠标",
                "primary_manual_document_id": 10,
            }
        ],
    )

    third = _plan_turn("不要键盘", metadata)
    third_args = third["decision"].tool_calls[0].arguments
    assert third_args["category"] == "鼠标和指针设备"
    assert third_args.get("keyword") != "键盘"
    _commit_planned_products(
        third["state"],
        third["decision"],
        [
            {
                "product_code": "2",
                "name": "MX Master 4",
                "category": "办公鼠标",
                "primary_manual_document_id": 10,
            }
        ],
    )

    fourth = _plan_turn("第一个，能充电么？", metadata)
    fourth_args = fourth["decision"].tool_calls[0].arguments
    assert fourth_args["product_code"] == "2"
    assert (
        fourth["state"]["customer_service_execution"]["pending_transaction"]["expected_result_type"]
        == "product_verification_for_manual"
    )


def test_category_switch_failure_does_not_restore_old_product_context() -> None:
    old_state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            filters={"category": "鼠标和指针设备"},
            active_batch=ProductCandidateBatch(
                batch_id="mouse-1",
                query="推荐鼠标",
                category="鼠标和指针设备",
                items=[
                    CandidateProduct(
                        product_code="M-1",
                        name="Mouse One",
                        category="鼠标和指针设备",
                        batch_id="mouse-1",
                        position=0,
                    )
                ],
            ),
            active_product_code="M-1",
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": old_state.model_dump(mode="json")},
    }

    planned = _plan_turn("给我推荐一个键盘", metadata)

    invalidated = CustomerServiceState.model_validate(metadata["customer_service"]["state"])
    assert invalidated.product == ProductContext()
    tool_call = planned["decision"].tool_calls[0]
    committed = CommitCoordinator().commit(
        agent_state=planned["state"],
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        result=ToolResult(
            name=tool_call.tool_name,
            success=False,
            error="database unavailable",
        ),
    )
    assert committed is False
    assert (
        CustomerServiceState.model_validate(metadata["customer_service"]["state"]).product
        == ProductContext()
    )


def test_single_product_result_is_automatically_selected() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }
    planned = _plan_turn("给我推荐一个鼠标", metadata)

    _commit_planned_products(
        planned["state"],
        planned["decision"],
        [
            {
                "product_code": "M-1",
                "name": "Mouse One",
                "category": "鼠标和指针设备",
            }
        ],
    )

    product = CustomerServiceState.model_validate(metadata["customer_service"]["state"]).product
    assert product.active_product_code == "M-1"
    assert product.active_batch is not None
    assert [item.product_code for item in product.active_batch.items] == ["M-1"]


def test_unspecified_recommendation_count_defaults_to_one() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("推荐个鼠标", metadata)

    assert planned["decision"].tool_calls[0].arguments["page_size"] == 1


def test_explicit_recommendation_count_is_preserved() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("推荐两个鼠标", metadata)

    assert planned["decision"].tool_calls[0].arguments["page_size"] == 2


def test_old_category_catalog_question_automatically_queries_one_new_product() -> None:
    keyboard = CustomerServiceState(
        product=ProductContext(
            active_category="键盘",
            filters={"category": "键盘"},
            active_batch=ProductCandidateBatch(
                batch_id="keyboard-1",
                query="推荐键盘",
                category="键盘",
                items=[
                    CandidateProduct(
                        product_code="K-1",
                        name="Keyboard One",
                        category="键盘",
                        batch_id="keyboard-1",
                        position=0,
                    )
                ],
            ),
            active_product_code="K-1",
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": keyboard.model_dump(mode="json")},
    }

    planned = _plan_turn("第一个鼠标多少钱", metadata)

    assert planned["decision"].tool_calls[0].tool_name == "recommend_products"
    assert planned["decision"].tool_calls[0].arguments["category"] == "鼠标和指针设备"
    assert planned["decision"].tool_calls[0].arguments["page_size"] == 1
    assert planned["decision"].tool_calls[0].arguments["knowledge_base_id"] == 1
    invalidated = CustomerServiceState.model_validate(metadata["customer_service"]["state"])
    assert invalidated.product == ProductContext()


def test_cross_category_manual_question_carries_scope_into_both_tools() -> None:
    keyboard = CustomerServiceState(
        product=ProductContext(
            active_category="键盘",
            filters={"category": "键盘"},
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": keyboard.model_dump(mode="json")},
    }
    planned = _plan_turn("第一个鼠标能连接蓝牙么", metadata)
    first_call = planned["decision"].tool_calls[0]

    assert first_call.tool_name == "recommend_products"
    assert first_call.arguments["knowledge_base_id"] == 1

    _commit_planned_products(
        planned["state"],
        planned["decision"],
        [
            {
                "product_code": "M-1",
                "name": "Mouse One",
                "category": "鼠标和指针设备",
                "primary_manual_document_id": 9,
            }
        ],
    )
    continued = asyncio.run(CustomerServiceStrategy().adecide(planned["state"]))
    second_call = continued.tool_calls[0]

    assert second_call.tool_name == "knowledge_search"
    assert second_call.arguments["knowledge_base_id"] == 1
    assert second_call.arguments["document_id"] == 9
    assert second_call.arguments["query"] == "Mouse One：第一个鼠标能连接蓝牙么"


def test_cross_category_product_answer_explains_automatic_requery() -> None:
    state = {
        "customer_service_execution": {
            "goal": {
                "semantic_frame": {
                    "intent": "product_fact_with_selection",
                    "slots": {"keyword": "鼠标"},
                }
            }
        },
        "observations": [
            {
                "success": True,
                "tool_name": "recommend_products",
                "raw_result": {
                    "items": [
                        {
                            "product": {
                                "product_code": "M-1",
                                "name": "Mouse One",
                                "price": "100.00",
                                "currency": "CNY",
                            }
                        }
                    ]
                },
            }
        ],
    }

    answer = CustomerServicePresenter().present(state)

    assert "不确定您询问的是哪款鼠标" in answer
    assert "现在为您推荐以下鼠标" in answer
    assert "Mouse One" in answer


def test_llm_fallback_receives_role_dialogue_and_confirms_pending_query(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "CUSTOMER_SERVICE_QUERY_REWRITE_MODE", "hybrid")
    captured_request = None
    captured_config = None

    class ConfirmationLLM:
        def chat(self, request):
            nonlocal captured_request
            captured_request = request
            return SimpleNamespace(
                tool_calls=[SimpleNamespace(arguments={"rewritten_query": "请重新查询鼠标"})]
            )

    def get_llm(*, config=None):
        nonlocal captured_config
        captured_config = config
        return ConfirmationLLM()

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        get_llm,
    )
    state = CustomerServiceState(
        product=ProductContext(
            pending_query=PendingProductQuery(
                category="鼠标和指针设备",
                keyword="鼠标",
                requested_count=1,
                created_turn_id="turn-1",
            )
        )
    )
    messages = [
        {
            "role": "user",
            "content": "我想查询鼠标",
        },
        {
            "role": "assistant",
            "content": "需要我重新为您查询鼠标吗？",
        },
        {"role": "user", "content": "麻烦处理一下"},
    ]

    result = asyncio.run(understand(query="麻烦处理一下", state=state, messages=messages))

    assert result.llm_used is True
    assert result.rewritten_query == "请重新查询鼠标"
    assert result.frame.intent == "recommend_products"
    assert captured_request is not None
    assert captured_config is not None
    assert captured_config.model == "qwen3.7-flash-2026-07-15"
    rewrite_schema = captured_request.tools[0]["function"]["parameters"]
    assert set(rewrite_schema["properties"]) == {"rewritten_query"}
    roles = [message.role for message in captured_request.messages]
    assert "assistant" in roles
    assert any(
        message.role == "assistant" and "重新为您查询鼠标" in message.content
        for message in captured_request.messages
    )


def test_simple_product_availability_question_never_calls_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    result = asyncio.run(understand(query="有鼠标么", state=CustomerServiceState(), messages=[]))

    assert result.llm_used is False
    assert result.rewritten_query is None
    assert result.frame.intent == "recommend_products"
    assert result.frame.slots["keyword"] == "鼠标"


def test_capability_selection_recommends_capability_then_adapter_builds_tool(
    monkeypatch,
) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")
    calls = 0
    fallback_request = None
    requested_models: list[str] = []

    class FallbackLLM:
        def chat(self, request):
            nonlocal calls, fallback_request
            calls += 1
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            if is_fallback:
                fallback_request = request
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="search_products" if is_fallback else "emit_rewritten_query",
                        arguments=(
                            {"keyword": "键帽", "page_size": 1}
                            if is_fallback
                            else {"rewritten_query": "你家卖键帽吗"}
                        ),
                    )
                ]
            )

    def get_llm(*, config=None):
        requested_models.append(config.model)
        return FallbackLLM()

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        get_llm,
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("你家卖键帽么", metadata)

    assert calls == 2
    assert requested_models == ["qwen3.7-flash-2026-07-15", "qwen-turbo"]
    assert fallback_request is not None
    native_tools = {
        tool["function"]["name"]: tool["function"] for tool in fallback_request.tools
    }
    assert "select_capability" not in native_tools
    assert "search_products" in native_tools
    assert native_tools["search_products"]["parameters"]["required"] == ["keyword"]
    assert native_tools["get_product_catalog_detail"]["parameters"]["required"] == [
        "product_code"
    ]
    assert set(native_tools["check_product_feature"]["parameters"]["required"]) == {
        "product_code",
        "feature",
    }
    assert fallback_request.tool_choice == "required"
    assert fallback_request.parallel_tool_calls is False
    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "search_products"
    assert tool_call.arguments["keyword"] == "键帽"
    assert (
        planned["state"]["metadata"]["customer_service"]["execution_details"][
            "capability_selection"
        ]["suggestion"]["tool_name"]
        == "search_products"
    )
    details = planned["state"]["metadata"]["customer_service"]["execution_details"]
    assert details["llm_call_count"] == 2
    assert set(details["llm_stages"]) == {
        "query_rewrite",
        "intent_routing",
        "slot_extraction",
        "reference_interpretation",
        "capability_selection",
    }
    assert details["llm_stages"]["query_rewrite"]["shared_call_id"] is None
    for stage_name in (
        "intent_routing",
        "slot_extraction",
        "reference_interpretation",
        "capability_selection",
    ):
        stage = details["llm_stages"][stage_name]
        assert stage["executed"] is True
        assert stage["source"] == "shared_llm"
        assert stage["shared_call_id"] == "heavy_semantic_tool_call"


def test_llm_only_mode_bypasses_business_rules_and_uses_capability_selector(
    monkeypatch,
) -> None:
    requested_tools: list[str] = []

    class RoutedLLM:
        def chat(self, request):
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            selected_name = "search_products" if is_fallback else "emit_rewritten_query"
            requested_tools.append(selected_name)
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=selected_name,
                        arguments=(
                            {"keyword": "鼠标", "page_size": 1}
                            if is_fallback
                            else {"rewritten_query": "你家卖鼠标吗"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: RoutedLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("你家卖鼠标么", metadata)

    assert requested_tools == [
        "emit_rewritten_query",
        "search_products",
    ]
    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "search_products"
    assert tool_call.arguments["keyword"] == "鼠标"
    understanding = planned["state"]["metadata"]["customer_service"][
        "execution_details"
    ]["understanding"]
    assert understanding["rule_frame"]["intent"] == "other"
    assert understanding["capability_selector_required"] is True


def test_planner_counts_light_and_shared_heavy_llm_calls_once(monkeypatch) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")

    class TwoStageLLM:
        def chat(self, request):
            is_heavy = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="search_products" if is_heavy else "emit_rewritten_query",
                        arguments=(
                            {"keyword": "鼠标", "page_size": 1}
                            if is_heavy
                            else {"rewritten_query": "你家卖鼠标吗"}
                        ),
                    )
                ]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: TwoStageLLM(),
    )
    state = {
        "query": "你家卖鼠标么",
        "messages": [{"role": "user", "content": "你家卖鼠标么"}],
        "metadata": {
            "agent_id": CUSTOMER_SERVICE_AGENT_ID,
            "planner_strategy": "customer_service",
            "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
        },
    }

    result = asyncio.run(PlannerNode().acall(state))

    assert result["llm_call_count"] == 2
    assert result["metadata"]["customer_service"]["execution_details"][
        "llm_calls_accounted"
    ] is True


def test_capability_selection_none_returns_clarification(monkeypatch) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")
    calls = 0

    class NoCapabilityLLM:
        def chat(self, request):
            nonlocal calls
            calls += 1
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="clarify_request" if is_fallback else "emit_rewritten_query",
                        arguments=(
                            {} if is_fallback else {"rewritten_query": "随便处理一下"}
                        ),
                    )
                ]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: NoCapabilityLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("随便处理一下", metadata)

    assert calls == 2
    assert planned["decision"].tool_calls == []
    assert "请说明您要查询的商品、订单或具体问题" in str(planned["decision"].content)


def test_capability_contract_accepts_after_sales_draft_candidate_only() -> None:
    suggestion = CapabilitySuggestion.model_validate(
        {
            "tool_name": "create_after_sales_ticket",
            "arguments": {
                "phone_last4": "5678",
                "issue_type": "return",
                "issue_description": "商品无法正常使用，需要退货",
            },
            "response_mode": "after_sales_draft",
            "question_slots": {},
            "reason": "只生成售后草稿候选",
        }
    )

    assert suggestion.response_mode == "after_sales_draft"


def test_heavy_llm_can_propose_after_sales_draft_but_not_confirm(monkeypatch) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")
    captured_request = None

    class AfterSalesLLM:
        def chat(self, request):
            nonlocal captured_request
            is_heavy = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            if is_heavy:
                captured_request = request
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "create_after_sales_draft"
                            if is_heavy
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {
                                "phone_last4": "5678",
                                "issue_type": "return",
                                "issue_description": "商品无法正常使用，需要退货",
                            }
                            if is_heavy
                            else {
                                "rewritten_query": (
                                    "为当前订单申请退货，手机号后四位5678，"
                                    "商品无法正常使用"
                                )
                            }
                        ),
                    )
                ]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: AfterSalesLLM(),
    )
    business_state = CustomerServiceState(
        order=OrderContext(active_order_ref="ORDER-001")
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    planned = _plan_turn(
        "我要退货，手机号后四位5678，商品无法正常使用",
        metadata,
    )

    call = planned["decision"].tool_calls[0]
    assert call.tool_name == "create_after_sales_ticket"
    assert call.arguments["action"] == "draft"
    assert call.arguments["order_no"] == "ORDER-001"
    assert call.arguments["customer_phone_last4"] == "5678"
    assert captured_request is not None
    available_tools = {
        tool["function"]["name"] for tool in captured_request.tools
    }
    assert "create_after_sales_draft" in available_tools
    assert "confirm_after_sales" not in available_tools


def test_heavy_llm_handoff_candidate_still_uses_typed_command(monkeypatch) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")

    class HandoffLLM:
        def chat(self, request):
            is_heavy = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "request_human_handoff"
                            if is_heavy
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {
                                "phone_last4": "5678",
                                "reason": "customer_request",
                                "message": "需要人工客服协助",
                            }
                            if is_heavy
                            else {"rewritten_query": "当前订单需要人工客服协助"}
                        ),
                    )
                ]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: HandoffLLM(),
    )
    business_state = CustomerServiceState(
        order=OrderContext(active_order_ref="ORDER-001")
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    planned = _plan_turn("转人工，手机号后四位5678", metadata)

    call = planned["decision"].tool_calls[0]
    assert call.tool_name == "create_human_handoff"
    assert call.arguments["order_no"] == "ORDER-001"
    assert call.arguments["customer_phone_last4"] == "5678"


def test_capability_selector_rejects_unknown_real_tool_argument(monkeypatch) -> None:
    _set_formal_stage_modes(monkeypatch, "llm_only")
    calls = 0

    class InvalidArgumentLLM:
        def chat(self, request):
            nonlocal calls
            calls += 1
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="search_products" if is_fallback else "emit_rewritten_query",
                        arguments=(
                            {"product_name": "鼠标"}
                            if is_fallback
                            else {"rewritten_query": "你家卖鼠标吗"}
                        ),
                    )
                ]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: InvalidArgumentLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("你家卖鼠标么", metadata)

    assert calls == 2
    assert planned["decision"].tool_calls == []
    failure = planned["state"]["metadata"]["customer_service"]["execution_details"][
        "capability_selection"
    ]["failure_reason"]
    assert "extra_forbidden" in failure


def test_capability_selector_rejects_product_list_without_query_condition(
    monkeypatch,
) -> None:
    class MissingConditionLLM:
        def chat(self, request):
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name="search_products" if is_fallback else "emit_rewritten_query",
                        arguments=(
                            {} if is_fallback else {"rewritten_query": "你家卖键盘么"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: MissingConditionLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("你家卖键盘么", metadata)

    assert planned["decision"].tool_calls == []
    failure = planned["state"]["metadata"]["customer_service"]["execution_details"][
        "capability_selection"
    ]["failure_reason"]
    assert "Field required" in failure
    assert "keyword" in failure


def test_native_catalog_detail_rejects_missing_product_code_before_real_tool(
    monkeypatch,
) -> None:
    class MissingProductCodeLLM:
        def chat(self, request):
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "get_product_catalog_detail"
                            if is_fallback
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {} if is_fallback else {"rewritten_query": "键盘是机械键盘吗"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: MissingProductCodeLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("是机械键盘吗", metadata)

    assert planned["decision"].tool_calls == []
    failure = planned["state"]["metadata"]["customer_service"]["execution_details"][
        "capability_selection"
    ]["failure_reason"]
    assert "Field required" in failure
    assert "product_code" in failure


def test_llm_only_current_product_use_case_enters_manual_evidence_chain(
    monkeypatch,
) -> None:
    fallback_request = None
    batch = ProductCandidateBatch(
        batch_id="keyboard-1",
        query="你家卖键盘么",
        category="键盘",
        items=[
            CandidateProduct(
                product_code="3",
                name="G512 X 75",
                category="键盘",
                batch_id="keyboard-1",
                position=0,
                primary_manual_document_id=11,
            )
        ],
    )

    class UseCaseLLM:
        def chat(self, request):
            nonlocal fallback_request
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            if is_fallback:
                fallback_request = request
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "search_product_manual"
                            if is_fallback
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {
                                "product_code": "3",
                                "manual_question": "G512 X 75键盘能打游戏么",
                            }
                            if is_fallback
                            else {"rewritten_query": "G512 X 75键盘能打游戏么"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: UseCaseLLM(),
    )
    business_state = CustomerServiceState(
        product=ProductContext(
            active_category="键盘",
            filters={"keyword": "键盘"},
            active_batch=batch,
            active_product_code="3",
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    planned = _plan_turn("能打游戏么", metadata)

    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "search_products"
    assert tool_call.arguments["product_code"] == "3"
    assert "keyword" not in tool_call.arguments
    transaction = planned["state"]["customer_service_execution"]["pending_transaction"]
    assert transaction["expected_result_type"] == "product_verification_for_manual"
    assert fallback_request is not None
    available_tools = {
        tool["function"]["name"] for tool in fallback_request.tools
    }
    assert "search_product_manual" in available_tools
    assert "check_product_use_case" not in available_tools


def test_llm_only_manual_question_without_trusted_product_selects_then_uses_rag(
    monkeypatch,
) -> None:
    fallback_request = None

    class ManualSelectionLLM:
        def chat(self, request):
            nonlocal fallback_request
            is_heavy = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            if is_heavy:
                fallback_request = request
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "select_product_for_manual"
                            if is_heavy
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {
                                "keyword": "G304",
                                "manual_question": "G304能打游戏么",
                            }
                            if is_heavy
                            else {"rewritten_query": "G304能打游戏么"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: ManualSelectionLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("G304能打游戏么", metadata)

    call = planned["decision"].tool_calls[0]
    assert call.tool_name == "recommend_products"
    assert call.arguments["keyword"] == "G304"
    assert call.arguments["page_size"] == 1
    transaction = planned["state"]["customer_service_execution"]["pending_transaction"]
    assert transaction["expected_result_type"] == "product_selection_for_manual_fact"
    goal = planned["state"]["customer_service_execution"]["goal"]
    assert goal["semantic_frame"]["requires_manual_evidence"] is True
    assert goal["semantic_frame"]["question"] == "G304能打游戏么"
    assert fallback_request is not None
    tools = {
        tool["function"]["name"]: tool["function"]
        for tool in fallback_request.tools
    }
    schema = tools["select_product_for_manual"]["parameters"]
    assert set(schema["required"]) == {"keyword", "manual_question"}


def test_llm_only_manual_fact_uses_trusted_product_then_scoped_knowledge(
    monkeypatch,
) -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-1",
        query="你家卖鼠标么",
        category="鼠标和指针设备",
        items=[
            CandidateProduct(
                product_code="1",
                name="罗技G304",
                category="鼠标和指针设备",
                batch_id="mouse-1",
                position=0,
                primary_manual_document_id=9,
            )
        ],
    )

    class ManualLLM:
        def chat(self, request):
            is_fallback = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "search_product_manual"
                            if is_fallback
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {
                                "product_code": "1",
                                "manual_question": "罗技G304鼠标能连接蓝牙吗",
                            }
                            if is_fallback
                            else {"rewritten_query": "罗技G304鼠标能连接蓝牙吗"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: ManualLLM(),
    )
    business_state = CustomerServiceState(
        product=ProductContext(
            active_category="鼠标和指针设备",
            active_batch=batch,
            active_product_code="1",
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    planned = _plan_turn("它能连接蓝牙么", metadata)

    first_call = planned["decision"].tool_calls[0]
    assert first_call.tool_name == "search_products"
    assert first_call.arguments["product_code"] == "1"
    _commit_planned_products(
        planned["state"],
        planned["decision"],
        [
            {
                "product_code": "1",
                "name": "罗技G304",
                "category": "鼠标和指针设备",
                "primary_manual_document_id": 9,
            }
        ],
    )
    continued = asyncio.run(CustomerServiceStrategy().adecide(planned["state"]))

    second_call = continued.tool_calls[0]
    assert second_call.tool_name == "knowledge_search"
    assert second_call.arguments["document_id"] == 9
    assert second_call.arguments["knowledge_base_id"] == 1
    assert "罗技G304鼠标能连接蓝牙吗" in second_call.arguments["query"]


def test_missing_manual_evidence_escalates_on_second_same_failure() -> None:
    batch = ProductCandidateBatch(
        batch_id="mouse-1",
        query="推荐鼠标",
        items=[
            CandidateProduct(
                product_code="1",
                name="罗技G304",
                batch_id="mouse-1",
                position=0,
                primary_manual_document_id=9,
            )
        ],
    )
    business_state = CustomerServiceState(
        product=ProductContext(
            active_batch=batch,
            active_product_code="1",
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    def run_without_evidence() -> dict:
        planned = _plan_turn("它能连接蓝牙么", metadata)
        _commit_planned_products(
            planned["state"],
            planned["decision"],
            [
                {
                    "product_code": "1",
                    "name": "罗技G304",
                    "primary_manual_document_id": 9,
                }
            ],
        )
        continued = asyncio.run(CustomerServiceStrategy().adecide(planned["state"]))
        knowledge_call = continued.tool_calls[0]
        committed = CommitCoordinator().commit(
            agent_state=planned["state"],
            tool_name=knowledge_call.tool_name,
            arguments=knowledge_call.arguments,
            result=ToolResult(
                name="knowledge_search",
                success=True,
                result={"answer": "", "sources": [], "citations": []},
                metadata={"document_id": 9, "knowledge_base_id": 1},
            ),
        )
        assert committed is False
        return planned["state"]

    first_state = run_without_evidence()

    after_first = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    )
    assert after_first.pending_clarification is not None
    assert after_first.pending_clarification.kind == "missing_evidence"
    assert "证据" in first_state["final_answer"]

    second_state = run_without_evidence()

    after_second = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    )
    assert after_second.pending_clarification is None
    assert "人工" in second_state["final_answer"]


def test_open_product_keyword_is_forwarded_to_recommendation_tool() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("推荐个键帽", metadata)

    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "recommend_products"
    assert tool_call.arguments["keyword"] == "键帽"


def test_recommendation_without_query_condition_stops_before_tool() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("推荐一个", metadata)

    assert planned["decision"].tool_calls == []
    assert "请说明您需要推荐的商品名称、类型或具体需求" in str(planned["decision"].content)


def test_llm_rewrite_context_keeps_complete_turns_and_excludes_raw_query(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "CUSTOMER_SERVICE_QUERY_REWRITE_MODE", "hybrid")
    class RewriteLLM:
        def chat(self, request):
            return SimpleNamespace(
                tool_calls=[SimpleNamespace(arguments={"rewritten_query": "再推荐一个鼠标"})]
            )

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: RewriteLLM(),
    )
    messages = [
        {"role": "user", "content": "有鼠标么"},
        {"role": "assistant", "content": "1. 罗技G304"},
        {"role": "user", "content": "我想买个鼠标，你给我推荐一个"},
        {"role": "assistant", "content": "1. 罗技G304"},
        {"role": "user", "content": "别的呢"},
    ]

    result = asyncio.run(
        understand(
            query="别的呢",
            state=CustomerServiceState(),
            messages=messages,
        )
    )

    assert result.llm_context is not None
    assert [(message.role, message.content) for message in result.llm_context.recent_dialogue] == [
        ("user", "有鼠标么"),
        ("assistant", "1. 罗技G304"),
        ("user", "我想买个鼠标，你给我推荐一个"),
        ("assistant", "1. 罗技G304"),
    ]


@pytest.mark.parametrize("query", ["需要", "要", "查吧", "帮我查一下"])
def test_common_pending_query_confirmations_are_rule_understood(query: str) -> None:
    state = CustomerServiceState(
        product=ProductContext(
            pending_query=PendingProductQuery(
                category="鼠标和指针设备",
                keyword="鼠标",
                requested_count=1,
                created_turn_id="turn-1",
            )
        )
    )

    result = asyncio.run(understand(query=query, state=state, messages=[]))

    assert result.llm_used is False
    assert result.frame.intent == "confirm_pending_product_query"


def test_pending_query_confirmation_builds_product_tool_call() -> None:
    state = CustomerServiceState(
        product=ProductContext(
            pending_query=PendingProductQuery(
                category="鼠标和指针设备",
                keyword="鼠标",
                requested_count=1,
                created_turn_id="turn-1",
            )
        )
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": state.model_dump(mode="json")},
    }

    planned = _plan_turn("需要", metadata)

    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "recommend_products"
    assert tool_call.arguments["category"] == "鼠标和指针设备"
    assert tool_call.arguments["page_size"] == 1


def test_streaming_customer_service_final_never_calls_final_llm(monkeypatch) -> None:
    async def fail_collect(*args, **kwargs):
        raise AssertionError("customer service final must not call an LLM")

    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.collect_streaming_answer",
        fail_collect,
    )
    queue = asyncio.Queue()
    state = {
        "query": "推荐鼠标",
        "messages": [{"role": "user", "content": "推荐鼠标"}],
        "observations": [
            {
                "tool_name": "recommend_products",
                "success": True,
                "raw_result": {
                    "items": [
                        {
                            "product": {
                                "product_code": "M-1",
                                "name": "Mouse One",
                            }
                        }
                    ]
                },
            }
        ],
        "metadata": {
            "agent_id": CUSTOMER_SERVICE_AGENT_ID,
            "_agent_stream_answer_enabled": True,
            "_agent_stream_event_queue": queue,
        },
    }

    result = asyncio.run(FinalNode().acall(state))

    assert result["final_answer"] == "1. Mouse One"
    assert queue.get_nowait() == {
        "event": "answer_delta",
        "data": {"delta": "1. Mouse One"},
    }


def test_rule_only_customer_service_turn_does_not_increment_llm_count() -> None:
    state = {
        "query": "你好",
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {
            "agent_id": CUSTOMER_SERVICE_AGENT_ID,
            "planner_strategy": "customer_service",
            "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
        },
    }

    result = asyncio.run(PlannerNode().acall(state))

    assert result["llm_call_count"] == 0
    assert result["final_answer"]


def test_order_list_then_ordinal_logistics_uses_trusted_candidate() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }
    first = _plan_turn("我的订单", metadata)
    assert first["decision"].tool_calls[0].tool_name == "query_order"
    _commit_tool_result(
        first["state"],
        first["decision"],
        {
            "mode": "list",
            "items": [
                {"order_no": "2026****0001", "status": "delivered"},
                {"order_no": "2026****0002", "status": "cancelled"},
            ],
            "total": 2,
        },
    )

    second = _plan_turn("第一个订单的物流", metadata)
    assert second["decision"].tool_calls[0].tool_name == "query_logistics"
    assert second["decision"].tool_calls[0].arguments["order_ref"] == "2026****0001"


def test_explicit_order_is_verified_before_logistics() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }
    planned = _plan_turn("查询订单202607240001的物流", metadata)
    first_call = planned["decision"].tool_calls[0]
    assert first_call.tool_name == "query_order"
    assert first_call.arguments == {"order_ref": "202607240001"}
    assert (
        planned["state"]["customer_service_execution"]["pending_transaction"][
            "expected_result_type"
        ]
        == "order_verification_for_logistics"
    )

    _commit_tool_result(
        planned["state"],
        planned["decision"],
        {
            "mode": "detail",
            "order_no": "2026****0001",
            "status": "shipped",
        },
    )
    continued = asyncio.run(CustomerServiceStrategy().adecide(planned["state"]))
    second_call = continued.tool_calls[0]
    assert second_call.tool_name == "query_logistics"
    assert second_call.arguments == {"order_ref": "202607240001"}


def test_after_sales_draft_then_confirm_uses_strong_commands() -> None:
    state = CustomerServiceState(active_order_ref="202607240001")
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": state.model_dump(mode="json")},
    }
    draft = _plan_turn(
        "我要退货，手机号后四位5678，商品无法正常使用",
        metadata,
    )
    draft_args = draft["decision"].tool_calls[0].arguments
    assert draft_args["action"] == "draft"
    assert draft_args["order_no"] == "202607240001"
    assert draft_args["customer_phone_last4"] == "5678"
    _commit_tool_result(
        draft["state"],
        draft["decision"],
        {
            "status": "draft",
            "draft_id": "mock-draft-0123456789abcdef01234567",
            "operation_id": "mock-draft-0123456789abcdef01234567",
            "summary": "订单退货",
        },
    )

    confirm = _plan_turn("确认提交", metadata)
    confirm_args = confirm["decision"].tool_calls[0].arguments
    assert confirm_args == {
        "action": "confirm",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "draft_id": "mock-draft-0123456789abcdef01234567",
        "operation_id": "mock-draft-0123456789abcdef01234567",
        "confirmed": True,
    }


def test_interrupted_after_sales_requires_rearming_before_confirm() -> None:
    business_state = CustomerServiceState(
        pending_after_sales=PendingAfterSales(
            draft_id="mock-draft-0123456789abcdef01234567",
            operation_id="mock-draft-0123456789abcdef01234567",
            order_no="202607240001",
            customer_phone_last4="5678",
            status="PENDING_CONFIRMATION",
        ),
        dialog_focus=DialogFocus(
            active_domain="product",
            active_action="recommend_products",
            source_turn_id="turn-product",
        ),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }

    rearm = _plan_turn("确认", metadata)

    assert rearm["decision"].action == "final"
    assert "售后" in str(rearm["decision"].content)
    state_after_rearm = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    )
    assert state_after_rearm.pending_after_sales is not None
    assert state_after_rearm.dialog_focus.active_domain == "after_sales"
    assert state_after_rearm.dialog_focus.active_action == "awaiting_explicit_confirmation"

    confirm = _plan_turn("确认提交", metadata)

    assert confirm["decision"].tool_calls[0].tool_name == "create_after_sales_ticket"
    assert confirm["decision"].tool_calls[0].arguments["action"] == "confirm"


def test_formal_after_sales_same_conversation_concurrent_confirm_executes_once() -> None:
    operation_id = "mock-draft-0123456789abcdef01234567"
    business_state = CustomerServiceState(
        pending_after_sales=PendingAfterSales(
            draft_id=operation_id,
            operation_id=operation_id,
            order_no="202607240001",
            customer_phone_last4="5678",
            status="PENDING_CONFIRMATION",
            created_turn_id="draft-turn",
            version=1,
        ),
        dialog_focus=DialogFocus(
            active_domain="after_sales",
            active_action="awaiting_explicit_confirmation",
            source_turn_id="draft-turn",
        ),
    )
    compatibility_pending = business_state.pending_after_sales.model_dump(mode="json")
    states: list[dict[str, Any]] = []
    for index in range(2):
        metadata = {
            "agent_id": CUSTOMER_SERVICE_AGENT_ID,
            "runtime_turn_id": f"confirm-turn-{index}",
            "tool_allowlist": ["create_after_sales_ticket"],
            "customer_service": {
                "state": business_state.model_dump(mode="json"),
                CUSTOMER_SERVICE_PENDING_KEY: deepcopy(compatibility_pending),
            },
        }
        planned = _plan_turn("确认提交", metadata)
        planned["state"]["messages"] = [{"role": "user", "content": "确认提交"}]
        planned["state"]["pending_tool_calls"] = [
            item.model_dump(mode="json") for item in planned["decision"].tool_calls
        ]
        states.append(planned["state"])

    class ConcurrentConfirmExecutor:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.calls = 0

        def execute(self, tool_call):
            with self._lock:
                self.calls += 1
            return ToolResult(
                name=tool_call.name,
                success=True,
                result={"status": "confirmed"},
            )

    executor = ConcurrentConfirmExecutor()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: asyncio.run(ToolNode(executor).acall(item)),
                states,
            )
        )

    assert executor.calls == 1
    statuses = [result["tool_results"][0]["status"] for result in results]
    assert sorted(statuses) == ["blocked", "success"]


@pytest.mark.parametrize("success_saved_first", [True, False])
def test_formal_runtime_session_cas_never_resurrects_confirmed_pending(
    monkeypatch,
    success_saved_first: bool,
) -> None:
    lock = threading.RLock()
    operation_id = "mock-draft-0123456789abcdef01234567"
    pending_state = CustomerServiceState(
        pending_after_sales=PendingAfterSales(
            draft_id=operation_id,
            operation_id=operation_id,
            order_no="202607240001",
            customer_phone_last4="5678",
            status="PENDING_CONFIRMATION",
        )
    )
    pending_metadata = {
        "state": pending_state.model_dump(mode="json"),
        CUSTOMER_SERVICE_PENDING_KEY: pending_state.pending_after_sales.model_dump(
            mode="json"
        ),
    }
    saved = MemoryState(
        session_id="conversation:42",
        revision=1,
        session_metadata={"customer_service": deepcopy(pending_metadata)},
    )

    class FakeCASMemoryManager:
        provider = type("Provider", (), {"name": "fake"})()

        def load_session(self, _session_id: str) -> MemoryState | None:
            with lock:
                return saved.model_copy(deep=True)

        def compare_and_save_session(
            self,
            state: MemoryState,
            *,
            expected_revision: int,
        ) -> bool:
            nonlocal saved
            with lock:
                if saved.revision != expected_revision:
                    return False
                saved = state.model_copy(deep=True)
                return True

    monkeypatch.setattr(
        "backend.app.agents.langgraph.runtime.MemoryFactory.get_manager",
        lambda: FakeCASMemoryManager(),
    )
    runtime = LangGraphAgentRuntime(graph_app=object())

    def runtime_state(customer_service: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": "确认提交",
            "messages": [{"role": "user", "content": "确认提交"}],
            "metadata": {
                "agent_id": CUSTOMER_SERVICE_AGENT_ID,
                "runtime_turn_id": "confirm-turn",
                "customer_service": customer_service,
                "session": {"revision": 1},
            },
            "customer_service_execution": CustomerServiceExecution(
                turn_id="confirm-turn"
            ).model_dump(mode="json"),
        }

    stale = runtime_state(deepcopy(pending_metadata))
    confirmed = runtime_state(
        {
            "state": CustomerServiceState().model_dump(mode="json"),
            "last_confirmed_operation_id": operation_id,
        }
    )
    ordered_states = [confirmed, stale] if success_saved_first else [stale, confirmed]
    for state in ordered_states:
        runtime._save_session("conversation:42", state)

    customer_service = saved.session_metadata["customer_service"]
    assert CUSTOMER_SERVICE_PENDING_KEY not in customer_service
    assert customer_service["last_confirmed_operation_id"] == operation_id


def test_same_clarification_second_failure_escalates_and_clears_pending() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    first = _plan_turn("第二个呢", metadata)

    assert first["decision"].action == "final"
    after_first = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    )
    assert after_first.pending_clarification is not None
    assert after_first.pending_clarification.attempts == 1

    second = _plan_turn("第二个呢", metadata)

    assert second["decision"].action == "final"
    assert "人工" in str(second["decision"].content)
    after_second = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    )
    assert after_second.pending_clarification is None


def test_pending_clarification_safe_slots_resume_after_sales(monkeypatch) -> None:
    business_state = CustomerServiceState(
        order=OrderContext(active_order_ref="202607240001")
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": business_state.model_dump(mode="json")},
    }
    first = _plan_turn("我要退货，商品无法正常使用", metadata)
    assert first["decision"].action == "final"
    pending = CustomerServiceState.model_validate(
        metadata["customer_service"]["state"]
    ).pending_clarification
    assert pending is not None
    assert pending.resume_action is not None
    assert pending.resume_action.safe_slots["issue_description"] == (
        "我要退货，商品无法正常使用"
    )

    class ResumeLLM:
        def chat(self, request):
            is_heavy = request.metadata.get("purpose") == (
                "customer_service_capability_selection"
            )
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        name=(
                            "create_after_sales_draft"
                            if is_heavy
                            else "emit_rewritten_query"
                        ),
                        arguments=(
                            {"phone_last4": "5678"}
                            if is_heavy
                            else {"rewritten_query": "手机号后四位是5678"}
                        ),
                    )
                ]
            )

    _set_formal_stage_modes(monkeypatch, "llm_only")
    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: ResumeLLM(),
    )

    resumed = _plan_turn("手机号后四位5678", metadata)

    call = resumed["decision"].tool_calls[0]
    assert call.tool_name == "create_after_sales_ticket"
    assert call.arguments["action"] == "draft"
    assert call.arguments["order_no"] == "202607240001"
    assert call.arguments["customer_phone_last4"] == "5678"
    assert call.arguments["issue_description"] == "我要退货，商品无法正常使用"


def test_explicit_order_is_verified_before_after_sales_draft() -> None:
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }
    planned = _plan_turn(
        "订单202607240001要退货，手机号后四位5678，商品无法正常使用",
        metadata,
    )
    first_call = planned["decision"].tool_calls[0]
    assert first_call.tool_name == "query_order"
    assert (
        planned["state"]["customer_service_execution"]["pending_transaction"][
            "expected_result_type"
        ]
        == "order_verification_for_after_sales"
    )

    _commit_tool_result(
        planned["state"],
        planned["decision"],
        {
            "mode": "detail",
            "order_no": "2026****0001",
            "status": "delivered",
        },
    )
    continued = asyncio.run(CustomerServiceStrategy().adecide(planned["state"]))
    second_call = continued.tool_calls[0]
    assert second_call.tool_name == "create_after_sales_ticket"
    assert second_call.arguments["action"] == "draft"
    assert second_call.arguments["order_no"] == "202607240001"


def test_formal_after_sales_policy_rejects_tampered_confirmation() -> None:
    pending = {
        "draft_id": "mock-draft-0123456789abcdef01234567",
        "operation_id": "mock-draft-0123456789abcdef01234567",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "status": "PENDING_CONFIRMATION",
        "created_turn_id": "turn-before",
    }
    state = CustomerServiceState(pending_after_sales=pending)
    command = ConfirmAfterSalesCommand(
        order_no="202607240001",
        customer_phone_last4="5678",
        draft_id=pending["draft_id"],
        operation_id=pending["operation_id"],
    )
    agent_state = _agent_state_for_command(command, state)
    agent_state["conversation_id"] = 52
    result = evaluate_tool_policy(
        state=agent_state,
        tool_name="create_after_sales_ticket",
        arguments={
            "action": "confirm",
            "order_no": "202607240001",
            "customer_phone_last4": "0000",
            "draft_id": pending["draft_id"],
            "operation_id": pending["operation_id"],
            "confirmed": True,
        },
    )
    assert result is not None
    assert result.success is False
    assert result.metadata["reason"] == "after_sales_confirmation_mismatch"


def _plan_turn(query: str, metadata: dict) -> dict:
    state = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "metadata": metadata,
        "conversation_id": 52,
        "knowledge_base_id": 1,
        "allowed_knowledge_base_ids": [1],
    }
    decision = asyncio.run(CustomerServiceStrategy().adecide(state))
    return {"state": state, "decision": decision}


def _commit_planned_products(
    state: dict,
    decision,
    items: list[dict],
) -> None:
    tool_call = decision.tool_calls[0]
    committed = CommitCoordinator().commit(
        agent_state=state,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        result=ToolResult(
            name=tool_call.tool_name,
            success=True,
            result={"items": items, "total": len(items)},
        ),
    )
    assert committed is True


def _commit_tool_result(
    state: dict,
    decision,
    result: dict,
) -> None:
    tool_call = decision.tool_calls[0]
    committed = CommitCoordinator().commit(
        agent_state=state,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        result=ToolResult(
            name=tool_call.tool_name,
            success=True,
            result=result,
        ),
    )
    assert committed is True


class _ProductVerificationExecutor:
    def execute(self, _tool_call):
        return ToolResult(
            name="search_products",
            success=True,
            result={
                "items": [
                    {
                        "product_code": "P-1",
                        "name": "Product One",
                        "category": "鼠标和指针设备",
                        "primary_manual_document_id": 77,
                    }
                ]
            },
        )


class _KnowledgeExecutor:
    def execute(self, _tool_call):
        return ToolResult(
            name="knowledge_search",
            success=True,
            result={
                "answer": "P-1 支持蓝牙连接。",
                "sources": [{"document_id": 77}],
                "citations": [{"document_id": 77}],
                "metadata": {},
            },
            metadata={"document_id": 77, "knowledge_base_id": 9},
        )


def _agent_state_for(command: SearchProductsCommand) -> dict:
    return _agent_state_for_command(command, CustomerServiceState())


def _agent_state_for_command(command, state: CustomerServiceState) -> dict:
    tool_name, _ = CommandAdapter(_registry()).adapt(command)
    transaction = PendingTransaction(
        transaction_id="tx-1",
        turn_id="turn-1",
        sequence=1,
        command=command,
        tool_call_id="call-1",
        tool_name=tool_name,
        arguments_hash="hash",
        proposed_patch={},
        expected_result_type="products",
    )
    execution = CustomerServiceExecution(
        turn_id="turn-1",
        pending_transaction=transaction,
    )
    return {
        "query": "test",
        "metadata": {
            "agent_id": CUSTOMER_SERVICE_AGENT_ID,
            "customer_service": {"state": state.model_dump(mode="json")},
        },
        "customer_service_execution": execution.model_dump(mode="json"),
    }
