from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.planning import RetrievalConstraint


class DenseRetrieveStep(BaseRetrieverStep):
    def __init__(self, dense_retriever: DenseRetriever | None = None) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        plan = context.retrieval_plan
        if plan is not None and not plan.dense_enabled:
            context.dense_chunks = []
            context.metadata["dense_total"] = 0
            context.metadata["dense_disabled_by_plan"] = True
            return context
        query = HybridRetrieveQuery(
            query=context.active_query,
            knowledge_base_id=context.knowledge_base_id,
            top_k=(
                context.retrieval_strategy.global_budget
                if context.retrieval_strategy is not None
                else context.top_k
            ),
            score_threshold=context.score_threshold,
            metadata_filter=context.metadata_filter,
            constraints=plan.constraints if plan is not None else [],
        )
        if plan is not None and plan.constraints:
            constraint_scope = context.metadata.setdefault("constraint_scope", {})
            constraint_scope["dense_scope"] = (
                "metadata_constraints" if plan.use_structure_filter else "knowledge_base"
            )
            constraint_scope["dense_constraint_count"] = len(
                [constraint for constraint in plan.constraints if constraint.applied]
            )
        chunks = self._retrieve_with_strategy(context, query)
        if not chunks and plan is not None and plan.use_structure_filter:
            chunks = self.dense_retriever.retrieve(query.model_copy(update={"constraints": []}))
            plan.fallback_used = True
            plan.fallback_reason = "dense_constraint_no_match"
            planning_metadata = context.metadata.setdefault("retrieval_planning", {})
            planning_metadata["fallback_used"] = True
            planning_metadata["fallback_reason"] = "dense_constraint_no_match"
            constraint_scope = context.metadata.setdefault("constraint_scope", {})
            constraint_scope["dense_fallback_used"] = True
            constraint_scope["dense_fallback_reason"] = "dense_constraint_no_match"
        if plan is not None and plan.constraints:
            constraint_scope = context.metadata.setdefault("constraint_scope", {})
            constraint_scope["dense_applied"] = bool(plan.use_structure_filter)
        candidate_ids = (
            context.retrieval_strategy.document_ids
            if context.retrieval_strategy is not None
            else plan.document_ids if plan is not None else []
        )
        if not candidate_ids and context.routing_result is None:
            candidate_ids = (
                context.auto_filter_result.candidate_document_ids
                if context.auto_filter_result is not None
                else []
            )
        if candidate_ids:
            allowed_ids = set(candidate_ids)
            before_count = len(chunks)
            chunks = [chunk for chunk in chunks if chunk.document_id in allowed_ids]
            rejected_count = before_count - len(chunks)
            retrieval_scope = context.metadata.setdefault("retrieval_scope", {})
            retrieval_scope.update(
                {
                    "candidate_document_ids": candidate_ids,
                    "dense_scope_applied": True,
                    "dense_rejected_count": rejected_count,
                }
            )
        context.dense_chunks = chunks
        context.metadata["dense_total"] = len(context.dense_chunks)
        return context

    def _retrieve_with_strategy(
        self,
        context: RetrieverPipelineContext,
        query: HybridRetrieveQuery,
    ):
        strategy = context.retrieval_strategy
        if strategy is None or strategy.strategy != "MULTI_DOCUMENT" or not strategy.document_ids:
            return self.dense_retriever.retrieve(query)

        per_document_budget = strategy.per_document_budget or strategy.global_budget
        chunks = []
        for document_id in strategy.document_ids:
            document_query = query.model_copy(
                update={
                    "top_k": per_document_budget,
                    "constraints": [
                        *query.constraints,
                        self._document_constraint(document_id),
                    ],
                }
            )
            document_chunks = [
                chunk
                for chunk in self.dense_retriever.retrieve(document_query)
                if chunk.document_id == document_id
            ][:per_document_budget]
            chunks.extend(document_chunks)
        return chunks[: strategy.global_budget]

    def _document_constraint(self, document_id: int) -> RetrievalConstraint:
        return RetrievalConstraint(
            field="document_id",
            operator="eq",
            value=document_id,
            confidence=1.0,
            source="retrieval_strategy",
            source_detail="per_document_budget",
            applied=True,
        )
