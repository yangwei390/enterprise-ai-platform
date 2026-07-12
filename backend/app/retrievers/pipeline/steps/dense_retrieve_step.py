from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


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
            top_k=context.top_k,
            score_threshold=context.score_threshold,
            metadata_filter=context.metadata_filter,
            constraints=plan.constraints if plan is not None else [],
        )
        chunks = self.dense_retriever.retrieve(query)
        if not chunks and plan is not None and plan.use_structure_filter:
            chunks = self.dense_retriever.retrieve(query.model_copy(update={"constraints": []}))
            plan.fallback_used = True
            plan.fallback_reason = "dense_constraint_no_match"
            context.metadata["retrieval_planning"]["fallback_used"] = True
            context.metadata["retrieval_planning"]["fallback_reason"] = (
                "dense_constraint_no_match"
            )
        if plan is not None and plan.constraints:
            constraint_scope = context.metadata.setdefault("constraint_scope", {})
            constraint_scope["dense_applied"] = bool(plan.use_structure_filter)
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
