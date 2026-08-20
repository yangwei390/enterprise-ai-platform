from __future__ import annotations

from enum import StrEnum

from backend.app.agents.customer_service_core.contracts import (
    CandidateProduct,
    CustomerServiceState,
    ReferenceExpression,
)
from pydantic import BaseModel, ConfigDict, Field


class ResolutionStatus(StrEnum):
    RESOLVED = "RESOLVED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ResolutionStatus
    product_codes: list[str] = Field(default_factory=list)
    verification_query: dict[str, str] = Field(default_factory=dict)
    candidates: list[CandidateProduct] = Field(default_factory=list)


def resolve_product_reference(
    reference: ReferenceExpression,
    state: CustomerServiceState,
) -> EntityResolution:
    batch = state.product.active_batch
    all_items = batch.items if batch is not None else []
    if reference.explicit_code:
        matched = [item for item in all_items if item.product_code == reference.explicit_code]
        if matched:
            return EntityResolution(
                status=ResolutionStatus.RESOLVED,
                product_codes=[matched[-1].product_code],
                candidates=[matched[-1]],
            )
        return EntityResolution(
            status=ResolutionStatus.VERIFICATION_REQUIRED,
            verification_query={"product_code": reference.explicit_code},
        )
    if reference.explicit_name:
        normalized = reference.explicit_name.casefold()
        matched = [item for item in all_items if normalized in item.name.casefold()]
        if len(matched) == 1:
            return EntityResolution(
                status=ResolutionStatus.RESOLVED,
                product_codes=[matched[0].product_code],
                candidates=matched,
            )
        if len(matched) > 1:
            return EntityResolution(
                status=ResolutionStatus.AMBIGUOUS,
                candidates=matched,
            )
        return EntityResolution(
            status=ResolutionStatus.VERIFICATION_REQUIRED,
            verification_query={"keyword": reference.explicit_name},
        )
    if reference.ordinal is not None:
        items = _active_batch_items(state, reference.category_hint)
        if items is None or reference.ordinal >= len(items):
            return EntityResolution(status=ResolutionStatus.NOT_FOUND)
        item = items[reference.ordinal]
        return EntityResolution(
            status=ResolutionStatus.RESOLVED,
            product_codes=[item.product_code],
            candidates=[item],
        )
    return EntityResolution(status=ResolutionStatus.AMBIGUOUS)


def _active_batch_items(
    state: CustomerServiceState,
    category_hint: str | None,
) -> list[CandidateProduct] | None:
    batch = state.product.active_batch
    if batch is None:
        return None
    if (
        category_hint is not None
        and state.product.active_category is not None
        and category_hint not in state.product.active_category
        and state.product.active_category not in category_hint
    ):
        return None
    return sorted(batch.items, key=lambda item: item.position)


def explicit_order_requires_verification(
    order_ref: str,
    state: CustomerServiceState,
) -> bool:
    batch = state.order.active_batch
    return not any(
        order_ref == item.order_ref
        for item in (batch.items if batch is not None else [])
    )
