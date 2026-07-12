from time import perf_counter

from backend.app.config.settings import settings
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.planning import RetrievalPlanningFactory


class RetrievalPlanningStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        started = perf_counter()
        candidate_document_ids = (
            context.auto_filter_result.candidate_document_ids
            if context.auto_filter_result is not None
            else []
        )
        planner = RetrievalPlanningFactory.get_planner()
        if not settings.RETRIEVAL_PLANNER_ENABLED:
            context.retrieval_plan = planner.fallback_plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                reason="disabled",
            )
            self._write_metadata(context, started)
            return context

        try:
            analysis = RetrievalPlanningFactory.get_analyzer().analyze(
                query=context.query,
                rewritten_query=context.active_query,
            )
            context.retrieval_plan = planner.plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                analysis=analysis,
            )
        except Exception as exc:
            if not settings.RETRIEVAL_PLANNER_FAIL_OPEN:
                raise
            context.retrieval_plan = planner.fallback_plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                reason=str(exc),
            )
        self._write_metadata(context, started)
        return context

    def _write_metadata(self, context: RetrieverPipelineContext, started: float) -> None:
        plan = context.retrieval_plan
        if plan is None:
            return
        duration_ms = round((perf_counter() - started) * 1000, 2)
        plan.metadata["planning_duration_ms"] = duration_ms
        context.metadata["retrieval_planning"] = {
            "enabled": settings.RETRIEVAL_PLANNER_ENABLED,
            "intent": plan.intent,
            "strategy": plan.strategy,
            "planner_source": plan.planner_source,
            "planning_duration_ms": duration_ms,
            "dense_enabled": plan.dense_enabled,
            "sparse_enabled": plan.sparse_enabled,
            "dense_weight": plan.dense_weight,
            "sparse_weight": plan.sparse_weight,
            "fallback_used": plan.fallback_used,
            "fallback_reason": plan.fallback_reason,
            "reason": plan.metadata.get("reason"),
        }
        context.metadata["constraints"] = [
            constraint.model_dump() for constraint in plan.constraints
        ]
        context.metadata["constraint_scope"] = (
            RetrievalPlanningFactory.get_constraint_engine().metadata(plan.constraints)
        )
