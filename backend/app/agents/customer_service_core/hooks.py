from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_AGENT_ID
from backend.app.agents.customer_service_core.after_sales_guard import (
    finalize_after_sales_confirmation,
    reserve_after_sales_confirmation,
)
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.tools.base import ToolResult


def is_customer_service_state(state: dict[str, Any]) -> bool:
    return state.get("metadata", {}).get("agent_id") == CUSTOMER_SERVICE_AGENT_ID


def prepare_tool_arguments(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not is_customer_service_state(state):
        return arguments
    execution = state.get("customer_service_execution")
    transaction = execution.get("pending_transaction") if isinstance(execution, dict) else None
    if not isinstance(transaction, dict) or transaction.get("tool_name") != tool_name:
        return arguments
    command = transaction.get("command")
    if not isinstance(command, dict):
        return arguments
    from backend.app.agents.customer_service_core.adapters import CommandAdapter
    from backend.app.agents.customer_service_core.contracts import PendingTransaction

    parsed = PendingTransaction.model_validate(transaction)
    adapted_name, validated = CommandAdapter().adapt(parsed.command)
    if adapted_name != tool_name:
        raise ValueError("command adapter returned a different tool")
    return validated


def commit_tool_result(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> bool | None:
    if not is_customer_service_state(state):
        return None
    if not isinstance(state.get("customer_service_execution"), dict):
        return False
    committed = CommitCoordinator().commit(
        agent_state=state,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
    )
    if tool_name == "create_after_sales_ticket" and arguments.get("action") == "confirm":
        operation_id = arguments.get("operation_id")
        if isinstance(operation_id, str):
            finalize_after_sales_confirmation(
                state=state,
                operation_id=operation_id,
                result=result,
            )
    return committed


def evaluate_tool_policy(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult | None:
    if not is_customer_service_state(state):
        return None
    execution = state.get("customer_service_execution")
    if not isinstance(execution, dict):
        return _blocked(tool_name, "customer_service_transaction_missing")
    transaction = execution.get("pending_transaction")
    if not isinstance(transaction, dict) or transaction.get("tool_name") != tool_name:
        return _blocked(tool_name, "customer_service_transaction_mismatch")
    if tool_name != "create_after_sales_ticket":
        return None
    if state.get("conversation_id") is None:
        return _blocked(tool_name, "after_sales_conversation_required")
    action = arguments.get("action", "draft")
    if action == "draft":
        return None
    if action != "confirm":
        return _blocked(tool_name, "unsupported_after_sales_action")
    pending = (
        state.get("metadata", {})
        .get("customer_service", {})
        .get("state", {})
        .get("pending_after_sales")
    )
    if not isinstance(pending, dict):
        return _blocked(tool_name, "after_sales_confirmation_missing")
    expected = {
        "order_no": pending.get("order_no"),
        "customer_phone_last4": pending.get("customer_phone_last4"),
        "draft_id": pending.get("draft_id"),
        "operation_id": pending.get("operation_id"),
        "confirmed": True,
    }
    actual = {key: arguments.get(key) for key in expected}
    if actual != expected:
        return _blocked(tool_name, "after_sales_confirmation_mismatch")
    if pending.get("created_turn_id") == execution.get("turn_id"):
        return _blocked(tool_name, "after_sales_same_turn_confirm_blocked")
    reservation_error = reserve_after_sales_confirmation(
        state=state,
        pending=pending,
    )
    if reservation_error is not None:
        return _blocked(tool_name, reservation_error)
    return None


def after_observation(state: Any) -> str | None:
    if not is_customer_service_state(state):
        return None
    execution = state.get("customer_service_execution")
    phase = execution.get("phase") if isinstance(execution, dict) else None
    if phase == "CONTINUE":
        return "planner"
    if phase in {"READY_FOR_FINAL", "READY_FOR_CLARIFICATION", "FAILED"}:
        if phase == "FAILED" and not state.get("final_answer"):
            state["final_answer"] = "本次业务操作未成功，可信状态未发生变化。"
        return "final"
    return None


def present_final_answer(state: Any) -> str | None:
    if not is_customer_service_state(state):
        return None
    from backend.app.agents.customer_service_core.presenter import (
        CustomerServicePresenter,
    )

    return CustomerServicePresenter().present(state)


def _blocked(tool_name: str, reason: str) -> ToolResult:
    return ToolResult(
        name=tool_name,
        success=False,
        error="客服业务操作被安全策略阻止",
        metadata={
            "status": "blocked",
            "reason": reason,
            "error_type": "customer_service_policy_error",
        },
    )
