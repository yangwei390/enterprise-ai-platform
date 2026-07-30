from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    SemanticFrame,
    StatePreview,
)


def preview_product_filters(
    state: CustomerServiceState,
    explicit_slots: dict[str, Any],
) -> tuple[CustomerServiceState, dict[str, Any]]:
    """生成状态预览；调用方不得把返回值当成已提交状态。"""
    preview = state.model_copy(deep=True)
    filters = dict(state.product.filters)
    old_category = state.product.active_category
    new_category = explicit_slots.get("category")
    category_changed = (
        new_category is not None and old_category is not None and new_category != old_category
    )
    if category_changed:
        filters = {}
        preview.product.active_batch = None
        preview.product.active_product_code = None
        preview.product.last_question = None
        preview.product.pending_query = None
    for key, value in explicit_slots.items():
        if value is None:
            continue
        filters[key] = value
    preview.product.filters = filters
    if isinstance(new_category, str):
        preview.product.active_category = new_category
    return preview, {
        "product_filters": filters,
        "invalidate_product_context": category_changed,
    }


def reduce_state(
    state: CustomerServiceState,
    frame: SemanticFrame,
) -> StatePreview:
    explicit_slots = {
        key: value
        for key, value in frame.slots.items()
        if key
        in {
            "keyword",
            "category",
            "brand",
            "model",
            "price_min",
            "price_max",
            "required_features",
            "required_use_cases",
            "in_stock_only",
            "sale_status",
        }
        and value is not None
    }
    proposed, patch = preview_product_filters(state, explicit_slots)
    changes: list[dict[str, Any]] = []

    for slot in frame.slots.get("remove_filters", []):
        if slot in proposed.product.filters:
            proposed.product.filters.pop(slot, None)
            changes.append({"slot": slot, "operation": "REMOVE"})

    if frame.continuation:
        seen_codes = _recent_relevant_product_codes(
            state,
            category=proposed.product.active_category,
        )
        existing = [
            str(value)
            for value in proposed.product.filters.get("excluded_product_codes", [])
            if value
        ]
        proposed.product.filters["excluded_product_codes"] = list(
            dict.fromkeys([*existing, *seen_codes])
        )
        changes.append(
            {
                "slot": "excluded_product_codes",
                "operation": "SET",
                "value": proposed.product.filters["excluded_product_codes"],
            }
        )

    if frame.intent == "cancel" and proposed.pending_after_sales is not None:
        proposed.pending_after_sales = None
        changes.append(
            {
                "slot": "pending_after_sales",
                "operation": "REMOVE",
            }
        )

    patch["product_filters"] = proposed.product.filters
    if frame.intent == "cancel":
        patch["pending_after_sales"] = None
    return StatePreview(
        state_before=state.model_copy(deep=True),
        proposed_state=proposed,
        proposed_patch=patch,
        change_log=changes,
    )


def _recent_relevant_product_codes(
    state: CustomerServiceState,
    *,
    category: Any,
) -> list[str]:
    batch = state.product.active_batch
    if batch is None:
        return []
    return [
        item.product_code
        for item in batch.items
        if category is None
        or item.category is None
        or str(category) in item.category
        or item.category in str(category)
    ]
