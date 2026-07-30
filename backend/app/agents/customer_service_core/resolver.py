from __future__ import annotations

from backend.app.agents.customer_service_core.schemas import (
    CandidateHistoryEntry,
    ConversationDST,
    CustomerServiceDomain,
    CustomerServiceIntent,
    OrderAction,
    TargetCardinality,
    TargetResolution,
    TargetResolutionPolicy,
    TargetResolutionSource,
)
from backend.app.agents.customer_service_core.semantic_schemas import (
    FieldSource,
    TargetSemantics,
)
from backend.app.schemas.product import normalize_product_category


def resolve_target(
    *,
    intent: CustomerServiceIntent,
    semantics: TargetSemantics | None,
    dst: ConversationDST,
    action: OrderAction | None = None,
) -> TargetResolution:
    if intent in {
        CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        CustomerServiceIntent.PRODUCT_SEARCH,
    }:
        return TargetResolution()
    if intent in {
        CustomerServiceIntent.PRODUCT_REALTIME_FACT,
        CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        CustomerServiceIntent.PRODUCT_COMPARISON,
    }:
        return _resolve_product_target(intent, semantics, dst)
    if intent in {
        CustomerServiceIntent.ORDER_QUERY,
        CustomerServiceIntent.LOGISTICS_QUERY,
        CustomerServiceIntent.AFTER_SALES,
        CustomerServiceIntent.HUMAN_HANDOFF,
    }:
        return _resolve_order_target(intent, semantics, dst, action)
    return TargetResolution()


def _resolve_product_target(
    intent: CustomerServiceIntent,
    semantics: TargetSemantics | None,
    dst: ConversationDST,
) -> TargetResolution:
    state = dst.domains.get(CustomerServiceDomain.PRODUCT)
    if state is None:
        return TargetResolution(clarification_required=True)
    if intent == CustomerServiceIntent.PRODUCT_COMPARISON and semantics is None:
        return resolve_targets(
            candidate_ids=[item.ref for item in state.candidates],
            policy=TargetResolutionPolicy(cardinality=TargetCardinality.ALL),
        )
    if semantics is None:
        return resolve_targets(
            candidate_ids=[item.ref for item in state.candidates],
            active_id=state.active_ref,
            policy=TargetResolutionPolicy(
                cardinality=TargetCardinality.SINGLE,
                allow_active=True,
                allow_single_candidate=True,
            ),
        )
    if semantics.explicit_ref:
        explicit_refs = semantics.explicit_refs or [semantics.explicit_ref]
        trusted_refs = {item.ref for item in state.candidates}
        trusted_refs.update(item.ref for item in state.candidate_history)
        if semantics.explicit_ref_source != FieldSource.EXPLICIT and any(
            item not in trusted_refs for item in explicit_refs
        ):
            return TargetResolution(clarification_required=True)
        return TargetResolution(
            resolved_ids=explicit_refs,
            source=TargetResolutionSource.EXPLICIT,
        )
    pool = (
        _filter_by_category_history(
            state.candidate_history,
            semantics.category_hint,
        )
        if semantics.category_hint
        else [
            CandidateHistoryEntry(
                ref=item.ref,
                display_name=item.display_name,
                batch_id="current",
                position=item.position,
                **(item.model_extra or {}),
            )
            for item in state.candidates
        ]
    )
    if (
        semantics.category_hint
        and not pool
        and state.candidates
        and all((item.model_extra or {}).get("category") is None for item in state.candidates)
    ):
        pool = [
            CandidateHistoryEntry(
                ref=item.ref,
                display_name=item.display_name,
                batch_id="current",
                position=item.position,
                **(item.model_extra or {}),
            )
            for item in state.candidates
        ]
    if semantics.category_hint and not pool:
        return TargetResolution(clarification_required=True)
    if semantics.explicit_name:
        matches = [
            item.ref
            for item in pool
            if semantics.explicit_name.casefold()
            in {
                item.ref.casefold(),
                str(item.display_name or "").casefold(),
                str((item.model_extra or {}).get("model") or "").casefold(),
            }
        ]
        if len(matches) == 1:
            return TargetResolution(
                resolved_ids=matches,
                source=TargetResolutionSource.EXPLICIT,
            )
        return TargetResolution(clarification_required=True)
    ordinals = semantics.ordinals or ([semantics.ordinal] if semantics.ordinal is not None else [])
    if ordinals:
        indices = [len(pool) - 1 if item == -1 else item for item in ordinals]
        if any(index < 0 or index >= len(pool) for index in indices):
            return TargetResolution(
                clarification_required=True,
                out_of_range=True,
            )
        return TargetResolution(
            resolved_ids=[pool[index].ref for index in indices],
            source=TargetResolutionSource.ORDINAL,
        )
    return resolve_targets(
        candidate_ids=[item.ref for item in pool],
        active_id=state.active_ref if not semantics.category_hint else None,
        policy=TargetResolutionPolicy(
            cardinality=TargetCardinality.SINGLE,
            allow_active=not semantics.category_hint,
            allow_single_candidate=True,
        ),
    )


def _resolve_order_target(
    intent: CustomerServiceIntent,
    semantics: TargetSemantics | None,
    dst: ConversationDST,
    action: OrderAction | None,
) -> TargetResolution:
    state = dst.domains.get(CustomerServiceDomain.ORDER)
    if state is None:
        return TargetResolution(
            clarification_required=action
            not in {
                OrderAction.LIST,
                OrderAction.COUNT,
                OrderAction.COMPARE,
            }
            and intent
            in {
                CustomerServiceIntent.LOGISTICS_QUERY,
                CustomerServiceIntent.AFTER_SALES,
                CustomerServiceIntent.HUMAN_HANDOFF,
            }
        )
    if semantics is not None and semantics.explicit_ref:
        explicit_refs = semantics.explicit_refs or [semantics.explicit_ref]
        trusted_refs = {item.ref for item in state.candidates}
        if semantics.explicit_ref_source != FieldSource.EXPLICIT and any(
            item not in trusted_refs for item in explicit_refs
        ):
            return TargetResolution(clarification_required=True)
        return TargetResolution(
            resolved_ids=explicit_refs,
            source=TargetResolutionSource.EXPLICIT,
        )
    if intent in {
        CustomerServiceIntent.ORDER_QUERY,
        CustomerServiceIntent.LOGISTICS_QUERY,
    } and action in {OrderAction.LIST, OrderAction.COUNT, OrderAction.COMPARE}:
        return TargetResolution(
            resolved_ids=[item.ref for item in state.candidates[:5]],
            source=(
                TargetResolutionSource.ALL_CANDIDATES
                if state.candidates
                else TargetResolutionSource.UNRESOLVED
            ),
        )
    if intent == CustomerServiceIntent.ORDER_QUERY and action is None and semantics is None:
        return TargetResolution(
            resolved_ids=[item.ref for item in state.candidates[:5]],
            source=(
                TargetResolutionSource.ALL_CANDIDATES
                if state.candidates
                else TargetResolutionSource.UNRESOLVED
            ),
        )
    if semantics is not None and semantics.ordinal is not None:
        index = len(state.candidates) - 1 if semantics.ordinal == -1 else semantics.ordinal
        if index < 0 or index >= len(state.candidates):
            return TargetResolution(
                clarification_required=True,
                out_of_range=True,
            )
        return TargetResolution(
            resolved_ids=[state.candidates[index].ref],
            source=TargetResolutionSource.ORDINAL,
        )
    if action == OrderAction.LOGISTICS and semantics is None and state.active_ref is None:
        return TargetResolution()
    return resolve_targets(
        candidate_ids=[item.ref for item in state.candidates],
        active_id=state.active_ref,
        policy=TargetResolutionPolicy(
            cardinality=TargetCardinality.SINGLE,
            allow_active=True,
            allow_single_candidate=True,
        ),
    )


def _filter_by_category_history(
    history: list[CandidateHistoryEntry],
    category: str,
) -> list[CandidateHistoryEntry]:
    normalized = normalize_product_category(category)
    if normalized is None:
        return []
    matched = [
        item
        for item in history
        if item.category is not None and normalize_product_category(item.category) == normalized
    ]
    if not matched:
        return []
    return matched


def resolve_targets(
    *,
    candidate_ids: list[str],
    policy: TargetResolutionPolicy,
    explicit_ids: list[str] | None = None,
    ordinal_indices: list[int] | None = None,
    active_id: str | None = None,
) -> TargetResolution:
    if policy.cardinality == TargetCardinality.NONE:
        return TargetResolution()
    if ordinal_indices:
        if any(index < 0 or index >= len(candidate_ids) for index in ordinal_indices):
            return TargetResolution(
                clarification_required=True,
                out_of_range=True,
            )
    resolved: list[str] = []
    for value in explicit_ids or []:
        if value not in resolved:
            resolved.append(value)
    for index in ordinal_indices or []:
        value = candidate_ids[index]
        if value not in resolved:
            resolved.append(value)
    if resolved:
        return TargetResolution(
            resolved_ids=resolved[:5],
            source=(
                TargetResolutionSource.EXPLICIT if explicit_ids else TargetResolutionSource.ORDINAL
            ),
        )
    if policy.cardinality == TargetCardinality.ALL and candidate_ids:
        return TargetResolution(
            resolved_ids=list(candidate_ids[:5]),
            source=TargetResolutionSource.ALL_CANDIDATES,
        )
    if policy.allow_active and active_id:
        return TargetResolution(
            resolved_ids=[active_id],
            source=TargetResolutionSource.ACTIVE,
        )
    if policy.allow_single_candidate and len(candidate_ids) == 1:
        return TargetResolution(
            resolved_ids=[candidate_ids[0]],
            source=TargetResolutionSource.SINGLE_CANDIDATE,
        )
    return TargetResolution(
        clarification_required=bool(candidate_ids),
    )


def normalized_target_position(
    reference: str,
    candidate_count: int,
) -> int | None:
    position = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "top": 0,
        "former": 0,
    }.get(reference.strip().casefold())
    if reference.strip().casefold() in {"bottom", "latter"} and candidate_count:
        return candidate_count - 1
    return position
