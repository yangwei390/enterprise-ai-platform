from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    SemanticFrame,
    StatePreview,
)

_CATEGORY_STABLE_SLOTS = {
    "brand",
    "price_min",
    "price_max",
    "in_stock_only",
    "sale_status",
}
_CATEGORY_RELATED_SLOTS = {
    "model",
    "features",
    "required_features",
    "preferred_features",
    "use_cases",
    "required_use_cases",
    "preferred_use_cases",
}


def preview_product_filters(
    state: CustomerServiceState,
    explicit_slots: dict[str, Any],
) -> tuple[CustomerServiceState, dict[str, Any]]:
    """生成状态预览；调用方不得把返回值当成已提交状态。"""
    preview = state.model_copy(deep=True)
    filters = dict(state.filters)
    old_category = filters.get("category")
    new_category = explicit_slots.get("category")
    category_changed = (
        new_category is not None
        and old_category is not None
        and new_category != old_category
    )
    if category_changed:
        filters = {
            key: value
            for key, value in filters.items()
            if key in _CATEGORY_STABLE_SLOTS
        }
        filters.pop("keyword", None)
        for key in _CATEGORY_RELATED_SLOTS:
            filters.pop(key, None)
    for key, value in explicit_slots.items():
        if value is None:
            continue
        filters[key] = value
    preview.filters = filters
    return preview, {"filters": filters}


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
        if slot in proposed.filters:
            proposed.filters.pop(slot, None)
            changes.append({"slot": slot, "operation": "REMOVE"})

    if frame.continuation:
        seen_codes = _recent_relevant_product_codes(
            state,
            category=proposed.filters.get("category"),
        )
        existing = [
            str(value)
            for value in proposed.filters.get("excluded_product_codes", [])
            if value
        ]
        proposed.filters["excluded_product_codes"] = list(
            dict.fromkeys([*existing, *seen_codes])
        )
        changes.append(
            {
                "slot": "excluded_product_codes",
                "operation": "SET",
                "value": proposed.filters["excluded_product_codes"],
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

    patch: dict[str, Any] = {"filters": proposed.filters}
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
    for batch in reversed(state.candidate_batches):
        matched = [
            item.product_code
            for item in batch.items
            if category is None
            or item.category is None
            or str(category) in item.category
            or item.category in str(category)
        ]
        if matched:
            return matched
    return []
