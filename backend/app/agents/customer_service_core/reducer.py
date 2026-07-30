from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import CustomerServiceState

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
