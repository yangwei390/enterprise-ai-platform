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
    ProductQuestionFocus,
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
        requires_continuation = transaction.expected_result_type in {
            "product_verification_for_manual",
            "product_selection_for_manual_fact",
            "order_list_for_logistics",
            "order_list_for_after_sales",
        } or transaction.expected_result_type.startswith("order_verification_for_")
        execution.phase = (
            ExecutionPhase.CONTINUE if requires_continuation else ExecutionPhase.READY_FOR_FINAL
        )
        agent_state["customer_service_execution"] = execution.model_dump(mode="json")
        self._trace(agent_state, before, updated, result)
        return True

    def commit_preview(
        self,
        *,
        agent_state: dict[str, Any],
        state: CustomerServiceState,
    ) -> None:
        customer_service = agent_state.setdefault("metadata", {}).setdefault(
            "customer_service",
            {},
        )
        customer_service["state"] = state.model_dump(mode="json")
        customer_service.pop(CUSTOMER_SERVICE_PENDING_KEY, None)

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
            validated_items = [ProductItemResult.model_validate(item) for item in raw_items]
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
            if execution.pending_transaction.expected_result_type.startswith(
                "product_verification"
            ):
                execution.verified_products = items
                return
            category = arguments.get("category")
            if not isinstance(category, str):
                category = (
                    items[0].category
                    if items and all(item.category == items[0].category for item in items)
                    else state.product.active_category
                )
            state.product.active_batch = ProductCandidateBatch(
                batch_id=batch_id,
                query=execution.goal.raw_query if execution.goal else "",
                category=category,
                items=items,
            )
            state.product.active_category = category
            state.product.active_product_code = items[0].product_code if len(items) == 1 else None
            state.product.last_question = None
            state.product.pending_query = None
            if tool_name in {"search_products", "recommend_products"}:
                proposed_filters = execution.pending_transaction.proposed_patch.get(
                    "product_filters"
                )
                if isinstance(proposed_filters, dict):
                    state.product.filters = proposed_filters
            self._commit_question_focus(state, execution)
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
                if (
                    execution.pending_transaction is not None
                    and execution.pending_transaction.expected_result_type.startswith(
                        "order_verification_for_"
                    )
                ):
                    execution.verified_order_ref = order_ref
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
            self._commit_question_focus(state, execution)
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
                metadata.setdefault("customer_service", {})[CUSTOMER_SERVICE_PENDING_KEY] = (
                    pending.model_dump(mode="json")
                )
            else:
                state.pending_after_sales = None
                metadata.setdefault("customer_service", {}).pop(
                    CUSTOMER_SERVICE_PENDING_KEY,
                    None,
                )

    @staticmethod
    def _commit_question_focus(
        state: CustomerServiceState,
        execution: CustomerServiceExecution,
    ) -> None:
        frame = execution.goal.semantic_frame if execution.goal is not None else None
        batch = state.product.active_batch
        predicate = frame.slots.get("question_predicate") if frame is not None else None
        if (
            frame is None
            or frame.intent not in {"product_fact", "product_fact_with_selection"}
            or batch is None
            or predicate
            not in {
                "bluetooth_connectivity",
                "charging",
                "compatibility",
                "price",
                "features",
                "buttons",
            }
        ):
            return
        state.product.last_question = ProductQuestionFocus(
            predicate=predicate,
            batch_id=batch.batch_id,
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
        details = (
            agent_state.setdefault("metadata", {})
            .setdefault(
                "customer_service",
                {},
            )
            .setdefault("execution_details", {})
        )
        details["tool_result"] = result.model_dump(mode="json")
        details["state_before"] = before.model_dump(mode="json")
        details["state_commit"] = after.model_dump(mode="json")
