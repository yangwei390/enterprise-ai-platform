from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    DialogFocus,
)


def reconcile_visible_context(
    state: CustomerServiceState,
    messages: list[dict[str, Any]],
) -> tuple[CustomerServiceState, list[dict[str, Any]]]:
    """使依赖原始消息可见性的引用状态与当前消息窗口保持一致。"""
    visible_turn_ids = _visible_turn_ids(messages)
    if not visible_turn_ids:
        return state, []

    reconciled = state.model_copy(deep=True)
    changes: list[dict[str, Any]] = []
    product_batch = reconciled.product.active_batch
    if (
        product_batch is not None
        and product_batch.source_turn_id is not None
        and product_batch.source_turn_id not in visible_turn_ids
    ):
        expired_batch_id = product_batch.batch_id
        reconciled.product.active_batch = None
        reconciled.product.active_product_code = None
        reconciled.product.last_question = None
        if reconciled.dialog_focus.active_batch_id == expired_batch_id:
            reconciled.dialog_focus = DialogFocus()
        changes.append(
            {
                "domain": "product",
                "reason": "source_turn_not_visible",
                "source_turn_id": product_batch.source_turn_id,
            }
        )

    order_batch = reconciled.order.active_batch
    if (
        order_batch is not None
        and order_batch.source_turn_id is not None
        and order_batch.source_turn_id not in visible_turn_ids
    ):
        expired_batch_id = order_batch.batch_id
        reconciled.order.active_batch = None
        reconciled.order.active_order_ref = None
        if reconciled.dialog_focus.active_batch_id == expired_batch_id:
            reconciled.dialog_focus = DialogFocus()
        changes.append(
            {
                "domain": "order",
                "reason": "source_turn_not_visible",
                "source_turn_id": order_batch.source_turn_id,
            }
        )

    question_focus = reconciled.product.last_question
    if (
        question_focus is not None
        and question_focus.source_turn_id is not None
        and question_focus.source_turn_id not in visible_turn_ids
    ):
        reconciled.product.last_question = None
        reconciled.dialog_focus.last_successful_predicate = None
        changes.append(
            {
                "domain": "manual",
                "reason": "question_turn_not_visible",
                "source_turn_id": question_focus.source_turn_id,
            }
        )
    return reconciled, changes


def _visible_turn_ids(messages: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        turn_id = message.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            metadata = message.get("metadata")
            turn_id = metadata.get("turn_id") if isinstance(metadata, dict) else None
        if isinstance(turn_id, str) and turn_id:
            result.add(turn_id)
    return result
