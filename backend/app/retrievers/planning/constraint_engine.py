from typing import Any

from backend.app.retrievers.planning.constraint_registry import ConstraintRegistry
from backend.app.retrievers.planning.schemas import RetrievalConstraint
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText, MatchValue, Range


class ConstraintEngine:
    top_level_payload_fields = {"document_id", "knowledge_base_id"}

    def __init__(self, registry: ConstraintRegistry) -> None:
        self.registry = registry

    def accepted_constraints(
        self,
        constraints: list[RetrievalConstraint],
    ) -> list[RetrievalConstraint]:
        return [
            constraint
            for constraint in constraints
            if constraint.applied and self.registry.get(constraint.field) is not None
        ]

    def to_qdrant_filter(
        self,
        constraints: list[RetrievalConstraint],
        *,
        base_filter: Filter | None = None,
    ) -> Filter | None:
        must: list[Any] = list(base_filter.must or []) if base_filter is not None else []
        for constraint in self.accepted_constraints(constraints):
            key = self._payload_key(constraint.field)
            if constraint.operator == "eq":
                must.append(
                    FieldCondition(key=key, match=MatchValue(value=constraint.value))
                )
            elif constraint.operator == "in":
                values = (
                    constraint.value
                    if isinstance(constraint.value, list)
                    else [constraint.value]
                )
                must.append(FieldCondition(key=key, match=MatchAny(any=values)))
            elif constraint.operator == "contains":
                must.append(
                    FieldCondition(key=key, match=MatchText(text=str(constraint.value)))
                )
            elif constraint.operator == "prefix":
                must.append(
                    FieldCondition(key=key, match=MatchText(text=str(constraint.value)))
                )
            elif constraint.operator == "range":
                range_filter = self._range_filter(constraint.value)
                if range_filter is not None:
                    must.append(FieldCondition(key=key, range=range_filter))
        if not must:
            return None
        return Filter(must=must)

    def matches_metadata(
        self,
        metadata: dict,
        constraints: list[RetrievalConstraint],
        *,
        document_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> bool:
        for constraint in self.accepted_constraints(constraints):
            value = self._value_for_field(
                metadata,
                constraint.field,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
            )
            if not self._matches(value, constraint.operator, constraint.value):
                return False
        return True

    def metadata(self, constraints: list[RetrievalConstraint]) -> dict:
        accepted = self.accepted_constraints(constraints)
        return {
            "registered_fields": [
                definition.field for definition in self.registry.list_fields()
            ],
            "accepted_count": len(accepted),
            "rejected_count": len(constraints) - len(accepted),
            "constraints": [constraint.model_dump() for constraint in constraints],
        }

    def _payload_key(self, field: str) -> str:
        if field in self.top_level_payload_fields:
            return field
        return f"metadata.{field}"

    def _value_for_field(
        self,
        metadata: dict,
        field: str,
        *,
        document_id: int | None,
        knowledge_base_id: int | None,
    ) -> Any:
        if field == "document_id":
            return document_id
        if field == "knowledge_base_id":
            return knowledge_base_id
        return metadata.get(field)

    def _matches(self, actual: Any, operator: str, expected: Any) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "in":
            values = expected if isinstance(expected, list) else [expected]
            return actual in values
        if operator == "contains":
            if isinstance(actual, list):
                return any(str(expected) in str(item) for item in actual)
            if isinstance(actual, str):
                return str(expected) in actual
            return False
        if operator == "prefix":
            if isinstance(actual, list):
                return any(str(item).startswith(str(expected)) for item in actual)
            if isinstance(actual, str):
                return actual.startswith(str(expected))
            return False
        if operator == "range":
            bounds = self._range_bounds(expected)
            if bounds is None:
                return False
            if not isinstance(actual, int | float):
                return False
            lower, upper = bounds
            if lower is not None and actual < lower:
                return False
            if upper is not None and actual > upper:
                return False
            return True
        return False

    def _range_filter(self, value: Any) -> Range | None:
        bounds = self._range_bounds(value)
        if bounds is None:
            return None
        lower, upper = bounds
        return Range(gte=lower, lte=upper)

    def _range_bounds(self, value: Any) -> tuple[float | None, float | None] | None:
        if isinstance(value, dict):
            lower = value.get("gte", value.get("min"))
            upper = value.get("lte", value.get("max"))
            return lower, upper
        if isinstance(value, list | tuple) and len(value) == 2:
            return value[0], value[1]
        return None
