from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_AGENT_ID
from backend.app.agents.customer_service_core.adapters import (
    CommandAdapter,
)
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.agents.customer_service_core.contracts import (
    CandidateProduct,
    CompareProductsCommand,
    ConfirmAfterSalesCommand,
    CreateAfterSalesDraftCommand,
    CreateHumanHandoffCommand,
    CustomerServiceExecution,
    CustomerServiceState,
    ExecutionPhase,
    GoalSnapshot,
    KnowledgeSearchCommand,
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
from backend.app.agents.customer_service_core.read_only_fallback import (
    ReadOnlyToolSuggestion,
)
from backend.app.agents.customer_service_core.reducer import (
    preview_product_filters,
    reduce_state,
)
from backend.app.agents.customer_service_core.strategy import CustomerServiceStrategy
from backend.app.agents.customer_service_core.understanding import understand
from backend.app.agents.langgraph.budget import AgentExecutionBudget
from backend.app.agents.langgraph.nodes import FinalNode, ObservationNode, PlannerNode, ToolNode
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
            ),
        )
    )

    result = asyncio.run(understand(query="第一个呢", state=state, messages=[]))

    assert result.llm_used is False
    assert result.frame.intent == "product_fact"
    assert result.frame.question == "该商品支持蓝牙吗"
    assert result.frame.references[0].ordinal == 0


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


def test_read_only_llm_fallback_recommends_capability_then_adapter_builds_tool(
    monkeypatch,
) -> None:
    calls = 0
    fallback_request = None

    class FallbackLLM:
        def chat(self, request):
            nonlocal calls, fallback_request
            calls += 1
            tool_name = request.tools[0]["function"]["name"]
            if tool_name == "recommend_read_only_capability":
                fallback_request = request
            arguments = (
                {
                    "tool_name": "recommend_products",
                    "arguments": {"keyword": "键帽", "page_size": 1},
                    "reason": "用户询问是否销售键帽",
                }
                if tool_name == "recommend_read_only_capability"
                else {"rewritten_query": "你家卖键帽吗"}
            )
            return SimpleNamespace(tool_calls=[SimpleNamespace(arguments=arguments)])

    monkeypatch.setattr(
        "backend.app.agents.customer_service_core.understanding.LLMFactory.get_llm",
        lambda **_kwargs: FallbackLLM(),
    )
    metadata = {
        "agent_id": CUSTOMER_SERVICE_AGENT_ID,
        "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
    }

    planned = _plan_turn("你家卖键帽么", metadata)

    assert calls == 2
    assert fallback_request is not None
    catalog_message = fallback_request.messages[-1].content
    assert '"name": "recommend_products"' in catalog_message
    assert '"keyword"' in catalog_message
    tool_call = planned["decision"].tool_calls[0]
    assert tool_call.tool_name == "recommend_products"
    assert tool_call.arguments["keyword"] == "键帽"
    assert (
        planned["state"]["metadata"]["customer_service"]["execution_details"][
            "read_only_tool_fallback"
        ]["suggestion"]["tool_name"]
        == "recommend_products"
    )


def test_read_only_llm_fallback_none_returns_clarification(monkeypatch) -> None:
    calls = 0

    class NoCapabilityLLM:
        def chat(self, request):
            nonlocal calls
            calls += 1
            tool_name = request.tools[0]["function"]["name"]
            arguments = (
                {
                    "tool_name": "none",
                    "arguments": {},
                    "reason": "没有安全的只读能力",
                }
                if tool_name == "recommend_read_only_capability"
                else {"rewritten_query": "随便处理一下"}
            )
            return SimpleNamespace(tool_calls=[SimpleNamespace(arguments=arguments)])

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


def test_read_only_fallback_contract_rejects_write_capability() -> None:
    with pytest.raises(ValueError):
        ReadOnlyToolSuggestion.model_validate(
            {
                "tool_name": "create_after_sales_ticket",
                "arguments": {},
                "reason": "禁止写操作",
            }
        )


def test_read_only_fallback_rejects_unknown_real_tool_argument(monkeypatch) -> None:
    calls = 0

    class InvalidArgumentLLM:
        def chat(self, request):
            nonlocal calls
            calls += 1
            tool_name = request.tools[0]["function"]["name"]
            arguments = (
                {
                    "tool_name": "search_products",
                    "arguments": {"product_name": "鼠标"},
                    "reason": "错误参数名",
                }
                if tool_name == "recommend_read_only_capability"
                else {"rewritten_query": "你家卖鼠标吗"}
            )
            return SimpleNamespace(tool_calls=[SimpleNamespace(arguments=arguments)])

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
        "read_only_tool_fallback"
    ]["failure_reason"]
    assert "extra_forbidden" in failure


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
