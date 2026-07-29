from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.app.agents.customer_service_core.schemas import (
    CandidateHistoryEntry,
    CandidateRef,
    ConversationDST,
    CustomerServiceDomain,
    DialogStatus,
    DialogTarget,
    DomainState,
    ProductRequestConstraints,
    SlotValue,
)
from pydantic import ValidationError

_DST_KEY = "dst"


def load_dst(metadata: dict[str, Any]) -> ConversationDST:
    customer_service = metadata.setdefault("customer_service", {})
    stored = customer_service.get(_DST_KEY)
    if isinstance(stored, dict):
        try:
            return ConversationDST.model_validate(stored)
        except ValidationError:
            pass
    dst = _migrate_legacy_state(customer_service)
    customer_service[_DST_KEY] = dst.model_dump(mode="json")
    return dst


def save_dst(metadata: dict[str, Any], dst: ConversationDST) -> None:
    dst.revision += 1
    metadata.setdefault("customer_service", {})[_DST_KEY] = dst.model_dump(mode="json")


def mutate_dst(
    metadata: dict[str, Any],
    mutation: Callable[[ConversationDST], None],
) -> ConversationDST:
    dst = load_dst(metadata)
    mutation(dst)
    _validate_invariants(dst)
    save_dst(metadata, dst)
    return dst


def replace_domain_candidates(
    dst: ConversationDST,
    *,
    domain: CustomerServiceDomain,
    candidates: list[dict[str, Any]],
    active_ref: str | None,
    filters: dict[str, Any] | None = None,
) -> None:
    refs = [
        CandidateRef(
            ref=str(item["ref"]),
            display_name=(
                str(item["display_name"]) if item.get("display_name") is not None else None
            ),
            position=index,
            **{
                key: value
                for key, value in item.items()
                if key not in {"ref", "display_name", "position"}
            },
        )
        for index, item in enumerate(candidates, start=1)
        if item.get("ref") is not None
    ]
    if active_ref is not None and active_ref not in {item.ref for item in refs}:
        raise ValueError("active_ref must belong to domain candidates")
    state = dst.domains.setdefault(domain, DomainState())
    state.candidates = refs
    state.active_ref = active_ref
    if filters is not None:
        state.filters = dict(filters)
    if domain == dst.active_domain:
        dst.active_target = next(
            (
                DialogTarget(
                    domain=domain,
                    ref=item.ref,
                    display_name=item.display_name,
                )
                for item in refs
                if item.ref == active_ref
            ),
            None,
        )


def record_candidate_batch(
    dst: ConversationDST,
    *,
    domain: CustomerServiceDomain,
    candidates: list[dict[str, Any]],
    batch_id: str,
) -> None:
    state = dst.domains.setdefault(domain, DomainState())
    existing = {(item.batch_id, item.ref) for item in state.candidate_history}
    additions = [
        CandidateHistoryEntry(
            ref=str(item["ref"]),
            display_name=(
                str(item["display_name"]) if item.get("display_name") is not None else None
            ),
            category=(
                str(item["category"]) if item.get("category") is not None else None
            ),
            batch_id=batch_id,
            position=index,
            **{
                key: value
                for key, value in item.items()
                if key not in {"ref", "display_name", "category", "batch_id", "position"}
            },
        )
        for index, item in enumerate(candidates, start=1)
        if item.get("ref") is not None
        and (batch_id, str(item["ref"])) not in existing
    ]
    state.candidate_history = [*state.candidate_history, *additions][-500:]


def set_status(dst: ConversationDST, status: DialogStatus) -> None:
    dst.status = status


def _validate_invariants(dst: ConversationDST) -> None:
    if dst.active_target is not None:
        domain_state = dst.domains.get(dst.active_target.domain)
        if domain_state is None or dst.active_target.ref not in {
            item.ref for item in domain_state.candidates
        }:
            raise ValueError("active_target must belong to trusted candidates")
    dst.missing_slots = [
        name
        for name in dst.required_slots
        if name not in dst.slots or not dst.slots[name].validated or dst.slots[name].value is None
    ]


def _migrate_legacy_state(customer_service: dict[str, Any]) -> ConversationDST:
    dst = ConversationDST()
    product_candidates = customer_service.get("recommendation_list")
    if not isinstance(product_candidates, list):
        context = customer_service.get("product_context")
        product_candidates = context.get("candidates", []) if isinstance(context, dict) else []
    product_refs = [
        {
            "ref": item["product_code"],
            "display_name": item.get("name") or item.get("model"),
            **item,
        }
        for item in product_candidates
        if isinstance(item, dict) and isinstance(item.get("product_code"), str)
    ]
    active_product = customer_service.get("active_product_code")
    if not isinstance(active_product, str):
        active_product = None
    replace_domain_candidates(
        dst,
        domain=CustomerServiceDomain.PRODUCT,
        candidates=product_refs,
        active_ref=(
            active_product if active_product in {item["ref"] for item in product_refs} else None
        ),
        filters=(
            customer_service.get("product_filters")
            if isinstance(customer_service.get("product_filters"), dict)
            else {}
        ),
    )
    product_domain = dst.domains[CustomerServiceDomain.PRODUCT]
    for name, value in product_domain.filters.items():
        if (
            name in ProductRequestConstraints.model_fields
            and value not in (None, "", [])
        ):
            dst.slots[name] = SlotValue(
                value=value,
                source="legacy_migration",
                validated=True,
            )
    seen_product_refs = customer_service.get("recommended_product_codes")
    product_domain.seen_refs = [
        ref for ref in seen_product_refs or [] if isinstance(ref, str) and ref
    ][-100:]
    record_candidate_batch(
        dst,
        domain=CustomerServiceDomain.PRODUCT,
        candidates=product_refs,
        batch_id="legacy_migration",
    )
    order_candidates = customer_service.get("order_candidates")
    order_refs = [
        {
            "ref": item["order_no"],
            "display_name": item.get("product_name"),
            **item,
        }
        for item in order_candidates or []
        if isinstance(item, dict) and isinstance(item.get("order_no"), str)
    ]
    active_order = customer_service.get("active_order_ref")
    if not isinstance(active_order, str):
        active_order = None
    replace_domain_candidates(
        dst,
        domain=CustomerServiceDomain.ORDER,
        candidates=order_refs,
        active_ref=(active_order if active_order in {item["ref"] for item in order_refs} else None),
    )
    record_candidate_batch(
        dst,
        domain=CustomerServiceDomain.ORDER,
        candidates=order_refs,
        batch_id="legacy_migration",
    )
    return dst
