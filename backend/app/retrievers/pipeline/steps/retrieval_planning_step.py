from time import perf_counter

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.planning import RetrievalPlanningFactory


class RetrievalPlanningStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        started = perf_counter()
        candidate_document_ids = self._candidate_document_ids(context)
        planner = RetrievalPlanningFactory.get_planner()
        if not settings.RETRIEVAL_PLANNER_ENABLED:
            context.retrieval_plan = planner.fallback_plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                reason="disabled",
            )
            self._attach_routing_metadata(context)
            self._write_metadata(context, started)
            return context

        try:
            analysis = RetrievalPlanningFactory.get_analyzer().analyze(
                query=context.query,
                rewritten_query=context.active_query,
                understanding=context.query_understanding,
            )
            context.retrieval_plan = planner.plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                analysis=analysis,
            )
            self._attach_routing_metadata(context)
        except Exception as exc:
            if not settings.RETRIEVAL_PLANNER_FAIL_OPEN:
                raise
            context.retrieval_plan = planner.fallback_plan(
                query=context.query,
                rewritten_query=context.active_query,
                candidate_document_ids=candidate_document_ids,
                reason=str(exc),
            )
            self._attach_routing_metadata(context)
        self._write_metadata(context, started)
        return context

    def _candidate_document_ids(self, context: RetrieverPipelineContext) -> list[int]:
        if context.routing_result is None:
            return (
                context.auto_filter_result.candidate_document_ids
                if context.auto_filter_result is not None
                else []
            )
        if context.routing_result.route_type in ("DOCUMENT", "MULTI_DOCUMENT"):
            return context.routing_result.target_document_ids
        return []

    def _attach_routing_metadata(self, context: RetrieverPipelineContext) -> None:
        if context.retrieval_plan is None or context.routing_result is None:
            return
        context.retrieval_plan.metadata["document_routing"] = (
            context.routing_result.model_dump()
        )

    def _write_metadata(self, context: RetrieverPipelineContext, started: float) -> None:
        plan = context.retrieval_plan
        if plan is None:
            return
        duration_ms = round((perf_counter() - started) * 1000, 2)
        plan.metadata["planning_duration_ms"] = duration_ms
        context.metadata["retrieval_planning"] = {
            "enabled": settings.RETRIEVAL_PLANNER_ENABLED,
            "query": plan.original_query,
            "rewritten_query": plan.rewritten_query,
            "intent": plan.intent,
            "strategy": plan.strategy,
            "planner_source": plan.planner_source,
            "planning_duration_ms": duration_ms,
            "constraint_source": [
                constraint.source_detail or constraint.source
                for constraint in plan.constraints
                if constraint.applied
            ],
            "metadata_constraints": [
                constraint.model_dump() for constraint in plan.constraints
            ],
            "dense_scope": (
                "metadata_constraints" if plan.use_structure_filter else "knowledge_base"
            ),
            "sparse_scope": (
                "metadata_constraints" if plan.use_structure_filter else "knowledge_base"
            ),
            "dense_enabled": plan.dense_enabled,
            "sparse_enabled": plan.sparse_enabled,
            "dense_weight": plan.dense_weight,
            "sparse_weight": plan.sparse_weight,
            "fallback_used": plan.fallback_used,
            "fallback_reason": plan.fallback_reason,
            "reason": plan.metadata.get("reason"),
            "document_routing_reused": "document_routing" in plan.metadata,
            "document_ids": plan.document_ids,
        }
        context.metadata["constraints"] = [
            constraint.model_dump() for constraint in plan.constraints
        ]
        context.metadata["constraint_scope"] = (
            RetrievalPlanningFactory.get_constraint_engine().metadata(plan.constraints)
        )
        logger.info(
            "Retrieval planning | query=%s | constraints=%s | dense_scope=%s | "
            "sparse_scope=%s | fallback=%s | planning_time_ms=%s",
            plan.original_query,
            context.metadata["retrieval_planning"]["metadata_constraints"],
            context.metadata["retrieval_planning"]["dense_scope"],
            context.metadata["retrieval_planning"]["sparse_scope"],
            plan.fallback_used,
            duration_ms,
        )
