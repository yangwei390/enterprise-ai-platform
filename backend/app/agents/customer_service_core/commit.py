from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_PENDING_KEY
from backend.app.agents.customer_service_core.contracts import (
    CandidateProduct,
    CustomerServiceExecution,
    CustomerServiceState,
    ExecutionPhase,
    PendingAfterSales,
    ProductCandidateBatch,
    TransactionStatus,
)
from backend.app.tools.base import ToolResult
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ProductResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    items: list[dict[str, Any]] = Field(default_factory=list)


class ProductItemResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    product_code: str
    name: str
    category: str | None = None
    primary_manual_document_id: int | None = None


class KnowledgeResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class CommitCoordinator:
    """唯一正式状态提交器。失败或契约不匹配时状态保持不变。"""

    def commit(
        self,
        *,
        agent_state: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> bool:
        execution = self._execution(agent_state)
        transaction = execution.pending_transaction
        if transaction is None or transaction.tool_name != tool_name:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = "missing_or_mismatched_transaction"
            agent_state["customer_service_execution"] = execution.model_dump(mode="json")
            return False
        if not result.success:
            transaction.status = TransactionStatus.REJECTED
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = result.error or "tool_failed"
            agent_state["customer_service_execution"] = execution.model_dump(mode="json")
            return False

        metadata = agent_state.setdefault("metadata", {})
        customer_service = metadata.setdefault("customer_service", {})
        before = CustomerServiceState.model_validate(customer_service.get("state", {}))
        updated = before.model_copy(deep=True)
        try:
            self._apply(
                updated,
                execution=execution,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                metadata=metadata,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            transaction.status = TransactionStatus.REJECTED
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = f"result_contract_failed:{exc}"
            agent_state["customer_service_execution"] = execution.model_dump(mode="json")
            return False

        customer_service["state"] = updated.model_dump(mode="json")
        transaction.status = TransactionStatus.COMMITTED
        execution.tool_count += 1
        execution.phase = (
            ExecutionPhase.CONTINUE
            if tool_name == "search_products"
            and transaction.expected_result_type == "product_verification"
            else ExecutionPhase.READY_FOR_FINAL
        )
        agent_state["customer_service_execution"] = execution.model_dump(mode="json")
        self._trace(agent_state, before, updated, result)
        return True

    def _apply(
        self,
        state: CustomerServiceState,
        *,
        execution: CustomerServiceExecution,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        metadata: dict[str, Any],
    ) -> None:
        if tool_name in {"search_products", "recommend_products", "compare_products"}:
            payload = ProductResult.model_validate(result.result)
            raw_items = [
                item.get("product", item) if isinstance(item, dict) else {}
                for item in payload.items
            ]
            validated_items = [
                ProductItemResult.model_validate(item) for item in raw_items
            ]
            if execution.pending_transaction is None:
                raise ValueError("pending transaction is required")
            batch_id = execution.pending_transaction.transaction_id
            items = [
                CandidateProduct(
                    product_code=item.product_code,
                    name=item.name,
                    category=item.category,
                    batch_id=batch_id,
                    position=index,
                    primary_manual_document_id=item.primary_manual_document_id,
                )
                for index, item in enumerate(validated_items)
            ]
            state.candidate_batches = [
                *state.candidate_batches[-7:],
                ProductCandidateBatch(
                    batch_id=batch_id,
                    query=execution.goal.raw_query if execution.goal else "",
                    category=arguments.get("category"),
                    items=items,
                ),
            ]
            if tool_name in {"search_products", "recommend_products"}:
                state.filters = {
                    key: value
                    for key, value in arguments.items()
                    if key
                    not in {"page", "page_size", "sort_by", "sort_order", "knowledge_base_id"}
                }
            return
        if tool_name == "query_order":
            if not isinstance(result.result, dict):
                raise TypeError("query_order result must be an object")
            if result.result.get("mode") == "list":
                items = result.result.get("items")
                if not isinstance(items, list):
                    raise TypeError("order list items must be a list")
                state.order_candidates = items
                state.active_order_ref = None
            else:
                order_ref = arguments.get("order_ref")
                if not isinstance(order_ref, str) or not order_ref:
                    raise ValueError("order detail requires verified order_ref")
                state.active_order_ref = order_ref
            return
        if tool_name == "query_logistics":
            if not isinstance(result.result, dict):
                raise TypeError("logistics result must be an object")
            state.active_order_ref = str(arguments["order_ref"])
            return
        if tool_name == "knowledge_search":
            knowledge = KnowledgeResult.model_validate(result.result)
            if not knowledge.sources or not knowledge.citations:
                raise ValueError("knowledge result has no evidence")
            expected_document_id = int(arguments["document_id"])
            actual_document_id = result.metadata.get("document_id")
            if actual_document_id != expected_document_id:
                raise ValueError("knowledge document scope mismatch")
            return
        if tool_name == "create_after_sales_ticket":
            if not isinstance(result.result, dict):
                raise TypeError("after-sales result must be an object")
            if arguments.get("action", "draft") == "draft":
                pending = PendingAfterSales(
                    draft_id=result.result["draft_id"],
                    operation_id=result.result["operation_id"],
                    order_no=arguments["order_no"],
                    customer_phone_last4=arguments["customer_phone_last4"],
                    status="PENDING_CONFIRMATION",
                    issue_type=arguments.get("issue_type"),
                    summary=result.result.get("summary"),
                    created_turn_id=execution.turn_id,
                    version=1,
                )
                state.pending_after_sales = pending
                metadata.setdefault("customer_service", {})[
                    CUSTOMER_SERVICE_PENDING_KEY
                ] = pending.model_dump(mode="json")
            else:
                state.pending_after_sales = None
                metadata.setdefault("customer_service", {}).pop(
                    CUSTOMER_SERVICE_PENDING_KEY,
                    None,
                )

    @staticmethod
    def _execution(agent_state: dict[str, Any]) -> CustomerServiceExecution:
        return CustomerServiceExecution.model_validate(
            agent_state.get("customer_service_execution")
        )

    @staticmethod
    def _trace(
        agent_state: dict[str, Any],
        before: CustomerServiceState,
        after: CustomerServiceState,
        result: ToolResult,
    ) -> None:
        details = agent_state.setdefault("metadata", {}).setdefault(
            "customer_service",
            {},
        ).setdefault("execution_details", {})
        details["tool_result"] = result.model_dump(mode="json")
        details["state_before"] = before.model_dump(mode="json")
        details["state_commit"] = after.model_dump(mode="json")
