from __future__ import annotations

from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_PENDING_KEY
from backend.app.agents.customer_service_core.contracts import (
    CandidateProduct,
    CustomerServiceExecution,
    CustomerServiceState,
    DialogFocus,
    ExecutionPhase,
    OrderCandidate,
    OrderCandidateBatch,
    PendingAfterSales,
    PendingClarification,
    ProductCandidateBatch,
    ProductQuestionFocus,
    ResumeAction,
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


class MissingEvidenceError(ValueError):
    pass


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
            if "pending_clarification" in transaction.proposed_patch:
                updated.pending_clarification = None
        except MissingEvidenceError:
            transaction.status = TransactionStatus.REJECTED
            self._record_missing_evidence(
                agent_state=agent_state,
                state=before,
                execution=execution,
            )
            agent_state["customer_service_execution"] = execution.model_dump(mode="json")
            return False
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
            if execution.pending_transaction.expected_result_type in {
                "product_catalog_detail",
                "product_feature_match",
            }:
                if len(items) != 1:
                    raise ValueError("product catalog fact requires one product")
                state.product.active_product_code = items[0].product_code
                state.product.last_question = None
                state.dialog_focus = DialogFocus(
                    active_domain="product",
                    active_action=execution.pending_transaction.expected_result_type,
                    active_batch_id=state.product.active_batch.batch_id
                    if state.product.active_batch is not None
                    else None,
                    source_turn_id=execution.turn_id,
                )
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
                source_turn_id=execution.turn_id,
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
            state.dialog_focus = DialogFocus(
                active_domain="product",
                active_action=execution.pending_transaction.expected_result_type,
                active_batch_id=batch_id,
                source_turn_id=execution.turn_id,
            )
            return
        if tool_name == "query_order":
            if not isinstance(result.result, dict):
                raise TypeError("query_order result must be an object")
            if result.result.get("mode") == "list":
                raw_items = result.result.get("items")
                if not isinstance(raw_items, list):
                    raise TypeError("order list items must be a list")
                if execution.pending_transaction is None:
                    raise ValueError("pending transaction is required")
                batch_id = execution.pending_transaction.transaction_id
                items = [
                    OrderCandidate.model_validate(
                        {
                            **item,
                            "batch_id": batch_id,
                            "position": index,
                        }
                    )
                    for index, item in enumerate(raw_items)
                    if isinstance(item, dict)
                ]
                if len(items) != len(raw_items):
                    raise TypeError("order list item must be an object")
                state.order.active_batch = OrderCandidateBatch(
                    batch_id=batch_id,
                    query=execution.goal.raw_query if execution.goal else "",
                    source_turn_id=execution.turn_id,
                    items=items,
                )
                state.order.active_order_ref = None
                state.dialog_focus = DialogFocus(
                    active_domain="order",
                    active_action="order_list",
                    active_batch_id=batch_id,
                    source_turn_id=execution.turn_id,
                )
            else:
                order_ref = arguments.get("order_ref")
                if not isinstance(order_ref, str) or not order_ref:
                    raise ValueError("order detail requires verified order_ref")
                state.order.active_order_ref = order_ref
                state.dialog_focus = DialogFocus(
                    active_domain="order",
                    active_action="order_detail",
                    active_batch_id=(
                        state.order.active_batch.batch_id
                        if state.order.active_batch is not None
                        else None
                    ),
                    source_turn_id=execution.turn_id,
                )
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
            state.order.active_order_ref = str(arguments["order_ref"])
            state.dialog_focus = DialogFocus(
                active_domain="logistics",
                active_action="logistics_status",
                active_batch_id=(
                    state.order.active_batch.batch_id
                    if state.order.active_batch is not None
                    else None
                ),
                source_turn_id=execution.turn_id,
            )
            return
        if tool_name == "knowledge_search":
            knowledge = KnowledgeResult.model_validate(result.result)
            if not knowledge.sources or not knowledge.citations:
                raise MissingEvidenceError("knowledge result has no evidence")
            expected_document_id = int(arguments["document_id"])
            actual_document_id = result.metadata.get("document_id")
            if actual_document_id != expected_document_id:
                raise ValueError("knowledge document scope mismatch")
            self._commit_question_focus(state, execution)
            state.dialog_focus = DialogFocus(
                active_domain="manual",
                active_action="manual_fact",
                active_batch_id=(
                    state.product.active_batch.batch_id
                    if state.product.active_batch is not None
                    else None
                ),
                last_successful_predicate=(
                    state.product.last_question.predicate
                    if state.product.last_question is not None
                    else None
                ),
                source_turn_id=execution.turn_id,
            )
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
                state.dialog_focus = DialogFocus(
                    active_domain="after_sales",
                    active_action="awaiting_explicit_confirmation",
                    source_turn_id=execution.turn_id,
                )
                metadata.setdefault("customer_service", {})[CUSTOMER_SERVICE_PENDING_KEY] = (
                    pending.model_dump(mode="json")
                )
            else:
                operation_id = arguments.get("operation_id")
                state.pending_after_sales = None
                state.dialog_focus = DialogFocus(
                    active_domain="after_sales",
                    active_action="confirmed",
                    source_turn_id=execution.turn_id,
                )
                metadata.setdefault("customer_service", {}).pop(
                    CUSTOMER_SERVICE_PENDING_KEY,
                    None,
                )
                if isinstance(operation_id, str) and operation_id:
                    metadata.setdefault("customer_service", {})[
                        "last_confirmed_operation_id"
                    ] = operation_id

    @staticmethod
    def _record_missing_evidence(
        *,
        agent_state: dict[str, Any],
        state: CustomerServiceState,
        execution: CustomerServiceExecution,
    ) -> None:
        frame = execution.goal.semantic_frame if execution.goal is not None else None
        question = (
            frame.question
            if frame is not None and isinstance(frame.question, str) and frame.question.strip()
            else execution.goal.raw_query
            if execution.goal is not None
            else "说明书问题"
        )
        target = f"manual:missing_evidence:{question.strip()}"
        existing = state.pending_clarification
        if (
            existing is not None
            and existing.kind == "missing_evidence"
            and existing.target_description == target
        ):
            state.pending_clarification = None
            agent_state["final_answer"] = (
                "连续两次都未找到足够的说明书证据，建议转人工客服继续核实。"
            )
        else:
            state.pending_clarification = PendingClarification(
                clarification_id=f"clarification_{uuid4().hex}",
                kind="missing_evidence",
                domain="manual",
                target_description=target,
                missing_fields=["evidence"],
                resume_action=ResumeAction(
                    action="product_fact",
                    domain="manual",
                    safe_slots={"question": question.strip()},
                ),
                attempts=1,
                created_turn_id=execution.turn_id,
                last_asked_turn_id=execution.turn_id,
            )
            agent_state["final_answer"] = (
                "暂时没有找到足够的说明书证据，无法可靠回答。您可以补充具体问题后再试。"
            )
        agent_state.setdefault("metadata", {}).setdefault("customer_service", {})[
            "state"
        ] = state.model_dump(mode="json")
        execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
        execution.failure_reason = "knowledge_evidence_missing"

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
                "use_case",
                "usage",
                "warranty",
            }
        ):
            return
        state.product.last_question = ProductQuestionFocus(
            predicate=predicate,
            batch_id=batch.batch_id,
            source_turn_id=execution.turn_id,
            source_question=frame.question,
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
