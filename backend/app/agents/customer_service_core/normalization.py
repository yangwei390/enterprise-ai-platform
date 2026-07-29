from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.schemas import (
    ProductConstraintOperations,
    SlotOperation,
    SlotUpdate,
)
from backend.app.schemas.product import normalize_product_category

_LIST_SLOTS = {
    "required_features",
    "preferred_features",
    "required_use_cases",
    "preferred_use_cases",
}
_PRICE_SLOTS = {"price_min", "price_max"}


def normalize_constraint_operations(
    operations: ProductConstraintOperations,
) -> ProductConstraintOperations:
    normalized: dict[str, SlotUpdate] = {}
    for name in ProductConstraintOperations.model_fields:
        update = getattr(operations, name)
        if update.op != SlotOperation.SET:
            normalized[name] = update
            continue
        normalized[name] = SlotUpdate(
            op=SlotOperation.SET,
            value=_normalize_slot_value(name, update.value),
        )
    return ProductConstraintOperations.model_validate(normalized)


def _normalize_slot_value(name: str, value: Any) -> Any:
    if name == "category":
        if not isinstance(value, str):
            raise ValueError("category 必须是字符串")
        normalized = normalize_product_category(value)
        if normalized is None:
            raise ValueError("category 不在标准商品大类中")
        return normalized
    if name in _PRICE_SLOTS:
        normalized = float(value)
        if normalized < 0:
            raise ValueError(f"{name} 不能小于 0")
        return normalized
    if name in _LIST_SLOTS:
        values = value if isinstance(value, list) else [value]
        result = [str(item).strip() for item in values if str(item).strip()]
        if not result:
            raise ValueError(f"{name} 不能为空")
        return result[:10]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()
