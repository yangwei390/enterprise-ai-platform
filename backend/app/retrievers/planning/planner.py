from backend.app.config.settings import settings
from backend.app.retrievers.planning.constraint_registry import ConstraintRegistry
from backend.app.retrievers.planning.schemas import (
    QueryAnalysisResult,
    RetrievalConstraint,
    RetrievalPlan,
    RetrievalStrategy,
)


class RetrievalPlanner:
    def __init__(self, registry: ConstraintRegistry) -> None:
        self.registry = registry

    def plan(
        self,
        *,
        query: str,
        rewritten_query: str,
        candidate_document_ids: list[int],
        analysis: QueryAnalysisResult,
    ) -> RetrievalPlan:
        accepted_constraints, rejected_constraints = self._validate_constraints(
            analysis.constraints
        )

        if accepted_constraints:
            strategy = self._strategy(settings.RETRIEVAL_STRUCTURED_STRATEGY)
            intent = "structured"
            dense_weight = settings.RETRIEVAL_DENSE_WEIGHT
            sparse_weight = settings.RETRIEVAL_SPARSE_WEIGHT
            reason = "structure_constraint_detected"
        elif analysis.intent == "lexical":
            strategy = "hybrid"
            intent = "lexical"
            dense_weight = 1 - settings.RETRIEVAL_SPARSE_HEAVY_WEIGHT
            sparse_weight = settings.RETRIEVAL_SPARSE_HEAVY_WEIGHT
            reason = "lexical_exact_token_detected"
        else:
            strategy = self._strategy(settings.RETRIEVAL_DEFAULT_STRATEGY)
            intent = analysis.intent
            dense_weight = settings.RETRIEVAL_DENSE_WEIGHT
            sparse_weight = settings.RETRIEVAL_SPARSE_WEIGHT
            reason = "default_hybrid"

        return RetrievalPlan(
            original_query=query,
            rewritten_query=rewritten_query,
            intent=intent,
            strategy=strategy,
            document_ids=candidate_document_ids,
            constraints=accepted_constraints + rejected_constraints,
            dense_enabled=strategy in ("dense", "hybrid", "structured_hybrid"),
            sparse_enabled=strategy in ("sparse", "hybrid", "structured_hybrid"),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            use_structure_filter=bool(accepted_constraints),
            planner_source="fast",
            metadata={
                "reason": reason,
                "accepted_constraints": len(accepted_constraints),
                "rejected_constraints": len(rejected_constraints),
                "analysis": analysis.metadata,
            },
        )

    def fallback_plan(
        self,
        *,
        query: str,
        rewritten_query: str,
        candidate_document_ids: list[int],
        reason: str,
    ) -> RetrievalPlan:
        return RetrievalPlan(
            original_query=query,
            rewritten_query=rewritten_query,
            intent="hybrid",
            strategy="hybrid",
            document_ids=candidate_document_ids,
            dense_enabled=True,
            sparse_enabled=True,
            dense_weight=settings.RETRIEVAL_DENSE_WEIGHT,
            sparse_weight=settings.RETRIEVAL_SPARSE_WEIGHT,
            planner_source="fast",
            fallback_used=True,
            fallback_reason=reason,
            metadata={"reason": "fail_open"},
        )

    def _validate_constraints(
        self,
        constraints: list[RetrievalConstraint],
    ) -> tuple[list[RetrievalConstraint], list[RetrievalConstraint]]:
        accepted: list[RetrievalConstraint] = []
        rejected: list[RetrievalConstraint] = []
        for constraint in constraints:
            definition = self.registry.get(constraint.field)
            if definition is None or not definition.enabled:
                rejected.append(
                    constraint.model_copy(update={"rejected_reason": "unknown_field"})
                )
                continue
            if constraint.operator not in definition.allowed_operators:
                rejected.append(
                    constraint.model_copy(update={"rejected_reason": "invalid_operator"})
                )
                continue
            if constraint.operator == "range":
                if not self._value_matches_range(constraint.value):
                    rejected.append(
                        constraint.model_copy(update={"rejected_reason": "type_mismatch"})
                    )
                    continue
                accepted.append(constraint.model_copy(update={"applied": True}))
                continue
            if not self._value_matches_type(constraint.value, definition.value_type):
                rejected.append(
                    constraint.model_copy(update={"rejected_reason": "type_mismatch"})
                )
                continue
            accepted.append(constraint.model_copy(update={"applied": True}))
        return accepted, rejected

    def _value_matches_type(self, value, value_type: str) -> bool:
        if value_type == "int":
            if isinstance(value, int):
                return True
            return isinstance(value, list) and all(isinstance(item, int) for item in value)
        if value_type == "str":
            if isinstance(value, str):
                return True
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        if value_type == "list":
            return isinstance(value, str | list)
        return False

    def _value_matches_range(self, value) -> bool:
        if isinstance(value, dict):
            lower = value.get("gte", value.get("min"))
            upper = value.get("lte", value.get("max"))
            return all(item is None or isinstance(item, int | float) for item in (lower, upper))
        if isinstance(value, list | tuple) and len(value) == 2:
            return all(item is None or isinstance(item, int | float) for item in value)
        return False

    def _strategy(self, value: str) -> RetrievalStrategy:
        if value in ("dense", "sparse", "hybrid", "structured_hybrid"):
            return value
        return "hybrid"
