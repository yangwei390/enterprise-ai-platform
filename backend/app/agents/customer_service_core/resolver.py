from __future__ import annotations

from backend.app.agents.customer_service_core.schemas import (
    TargetCardinality,
    TargetResolution,
    TargetResolutionPolicy,
    TargetResolutionSource,
)


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
