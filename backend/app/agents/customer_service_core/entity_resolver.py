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
    all_items = [
        item
        for batch in state.candidate_batches
        for item in batch.items
    ]
    if reference.explicit_code:
        matched = [
            item for item in all_items if item.product_code == reference.explicit_code
        ]
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
        matched = [
            item
            for item in all_items
            if normalized in item.name.casefold()
        ]
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
        batch = _recent_relevant_batch(state, reference.category_hint)
        if batch is None or reference.ordinal >= len(batch):
            return EntityResolution(status=ResolutionStatus.NOT_FOUND)
        item = batch[reference.ordinal]
        return EntityResolution(
            status=ResolutionStatus.RESOLVED,
            product_codes=[item.product_code],
            candidates=[item],
        )
    return EntityResolution(status=ResolutionStatus.AMBIGUOUS)


def _recent_relevant_batch(
    state: CustomerServiceState,
    category_hint: str | None,
) -> list[CandidateProduct] | None:
    for batch in reversed(state.candidate_batches):
        items = batch.items
        if category_hint is not None:
            items = [
                item
                for item in items
                if item.category is not None
                and (
                    category_hint in item.category
                    or item.category in category_hint
                )
            ]
        if items:
            return sorted(items, key=lambda item: item.position)
    return None


def explicit_order_requires_verification(
    order_ref: str,
    state: CustomerServiceState,
) -> bool:
    return not any(
        order_ref
        in {
            str(item.get("order_ref") or ""),
            str(item.get("order_no") or ""),
        }
        for item in state.order_candidates
    )
