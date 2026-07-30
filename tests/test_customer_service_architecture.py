from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_AGENT_ID
from backend.app.agents.customer_service_core.adapters import (
    CommandAdapter,
)
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.agents.customer_service_core.contracts import (
    AfterSalesCommand,
    CandidateProduct,
    CompareProductsCommand,
    CustomerServiceExecution,
    CustomerServiceState,
    ExecutionPhase,
    GoalSnapshot,
    HumanHandoffCommand,
    KnowledgeSearchCommand,
    PendingTransaction,
    ProductCandidateBatch,
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
from backend.app.agents.customer_service_core.reducer import preview_product_filters
from backend.app.agents.customer_service_core.strategy import CustomerServiceStrategy
from backend.app.agents.langgraph.budget import AgentExecutionBudget
from backend.app.agents.langgraph.nodes import FinalNode, ObservationNode, ToolNode
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
        AfterSalesCommand(
            arguments={
                "action": "draft",
                "order_no": "ORDER-001",
                "customer_phone_last4": "1234",
                "issue_type": "return",
                "issue_description": "商品无法正常使用",
            }
        ),
        HumanHandoffCommand(
            arguments={
                "order_no": "ORDER-001",
                "customer_phone_last4": "1234",
                "reason": "customer_request",
                "message": "需要人工协助",
            }
        ),
    ],
)
def test_every_command_validates_against_real_tool_schema(command) -> None:
    tool_name, arguments = CommandAdapter(_registry()).adapt(command)
    tool = _registry().get_tool(tool_name, require_enabled=True)
    assert tool is not None
    tool.args_schema.model_validate(arguments)


def test_category_switch_keeps_only_stable_slots_and_explicit_new_values() -> None:
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
    assert state.filters["category"] == "鼠标和指针设备"
    assert preview.filters == {
        "brand": "Logitech",
        "price_max": 500,
        "category": "键盘",
        "required_features": ["机械轴"],
        "keyword": "无线",
    }
    assert patch == {"filters": preview.filters}


def test_ordinal_resolves_only_inside_recent_relevant_batch() -> None:
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
    assert resolution.status == ResolutionStatus.RESOLVED
    assert resolution.product_codes == ["M-1"]


def test_explicit_entity_outside_pool_requires_tool_verification() -> None:
    resolution = resolve_product_reference(
        ReferenceExpression(text="P-404", explicit_code="P-404"),
        CustomerServiceState(),
    )
    assert resolution.status == ResolutionStatus.VERIFICATION_REQUIRED
    assert resolution.verification_query == {"product_code": "P-404"}


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
    execution = CustomerServiceExecution.model_validate(
        state["customer_service_execution"]
    )
    execution.phase = ExecutionPhase.WAITING_TOOL
    execution.goal = GoalSnapshot(raw_query="P-1 支持蓝牙吗")
    execution.pending_transaction.expected_result_type = "product_verification"
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

    state["pending_tool_calls"] = [
        item.model_dump(mode="json") for item in decision.tool_calls
    ]
    asyncio.run(ToolNode(_KnowledgeExecutor()).acall(state))
    asyncio.run(ObservationNode().acall(state))
    asyncio.run(FinalNode().acall(state))
    assert state["customer_service_execution"]["phase"] == "READY_FOR_FINAL"
    assert state["final_answer"] == "P-1 支持蓝牙连接。"
    assert state["tool_call_count"] == 2


def test_planning_state_does_not_copy_stream_runtime_objects() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    future = asyncio.get_event_loop_policy().new_event_loop().create_future()
    state = {
        "metadata": {
            "_agent_stream_event_queue": queue,
            "_agent_stream_future": future,
            "customer_service": {
                "state": CustomerServiceState().model_dump(mode="json")
            },
        }
    }

    planning_state = CustomerServiceStrategy._planning_state(state)

    assert planning_state["metadata"]["_agent_stream_event_queue"] is queue
    assert planning_state["metadata"]["_agent_stream_future"] is future
    assert planning_state["metadata"]["customer_service"] is not state["metadata"][
        "customer_service"
    ]
    future.get_loop().close()


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
    transaction = PendingTransaction(
        transaction_id="tx-1",
        turn_id="turn-1",
        sequence=1,
        command=command,
        tool_call_id="call-1",
        tool_name=command.kind,
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
            "customer_service": {"state": CustomerServiceState().model_dump(mode="json")},
        },
        "customer_service_execution": execution.model_dump(mode="json"),
    }
