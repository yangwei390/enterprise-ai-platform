from __future__ import annotations

from backend.app.agents.customer_service_core.compatibility import (
    CompatibilityResult,
    CompatibilityStatus,
)
from backend.app.agents.customer_service_core.fsm import (
    apply_request,
    product_constraints_from_dst,
)
from backend.app.agents.customer_service_core.normalization import (
    normalize_constraint_operations,
)
from backend.app.agents.customer_service_core.schemas import (
    ContextualizedRequest,
    ConversationDST,
    CustomerServiceDomain,
    CustomerServiceIntent,
    CustomerServiceSource,
    ProductConstraintOperations,
    ProductPayload,
    SlotOperation,
    SlotUpdate,
    SlotValue,
)


def _request(**updates: SlotUpdate) -> ContextualizedRequest:
    operations = normalize_constraint_operations(
        ProductConstraintOperations.model_validate(updates)
    )
    return ContextualizedRequest(
        raw_query="商品查询",
        rewritten_query="商品查询",
        domain=CustomerServiceDomain.PRODUCT,
        intent=CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        action="recommend_products",
        source=CustomerServiceSource.PRODUCT_CATALOG,
        payload=ProductPayload(constraint_operations=operations),
    )


def _slot(value: object) -> SlotValue:
    return SlotValue(value=value, source="test", validated=True)


def test_product_slots_merge_by_operation() -> None:
    dst = ConversationDST(
        slots={
            "category": _slot("鼠标和指针设备"),
            "brand": _slot("罗技"),
            "price_max": _slot(800),
        }
    )

    apply_request(
        dst,
        _request(
            category=SlotUpdate(op=SlotOperation.SET, value="keyboard"),
            brand=SlotUpdate(op=SlotOperation.KEEP),
            price_max=SlotUpdate(op=SlotOperation.REMOVE),
        ),
    )

    assert dst.slots["category"].value == "键盘"
    assert dst.slots["brand"].value == "罗技"
    assert "price_max" not in dst.slots


def test_unknown_compatibility_preserves_state_but_excludes_tool_constraint() -> None:
    dst = ConversationDST(
        slots={
            "category": _slot("鼠标和指针设备"),
            "required_use_cases": _slot(["游戏"]),
        }
    )

    apply_request(
        dst,
        _request(category=SlotUpdate(op=SlotOperation.SET, value="键盘")),
    )

    assert dst.slots["required_use_cases"].value == ["游戏"]
    assert (
        dst.suppressed_slots["required_use_cases"]
        == "category_compatibility_unknown"
    )
    assert product_constraints_from_dst(dst).required_use_cases == []
    assert any(
        change["slot"] == "required_use_cases"
        and change["action"] == "suppress"
        for change in dst.slot_change_log
    )


class _IncompatibleProvider:
    def evaluate(self, *, dst, category, slot, value) -> CompatibilityResult:
        return CompatibilityResult(
            status=CompatibilityStatus.INCOMPATIBLE,
            slot=slot,
            reason="catalog_incompatible",
            source="test_catalog",
        )


def test_incompatible_inherited_slot_is_cleared_with_reason() -> None:
    dst = ConversationDST(
        slots={
            "category": _slot("鼠标和指针设备"),
            "required_use_cases": _slot(["游戏"]),
        }
    )

    apply_request(
        dst,
        _request(category=SlotUpdate(op=SlotOperation.SET, value="键盘")),
        compatibility_provider=_IncompatibleProvider(),
    )

    assert "required_use_cases" not in dst.slots
    assert any(
        change == {
            "slot": "required_use_cases",
            "action": "clear",
            "reason": "catalog_incompatible",
            "revision": 1,
        }
        for change in dst.slot_change_log
    )
