from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_contract import CUSTOMER_SERVICE_AGENT_ID
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
        from backend.app.agents.customer_service import (
            prepare_customer_service_tool_arguments,
        )

        return prepare_customer_service_tool_arguments(
            state=state,
            tool_name=tool_name,
            arguments=arguments,
        )
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
        from backend.app.agents.customer_service import (
            update_customer_service_state_after_tool,
        )

        update_customer_service_state_after_tool(
            state=state,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        return None
    return CommitCoordinator().commit(
        agent_state=state,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
    )
