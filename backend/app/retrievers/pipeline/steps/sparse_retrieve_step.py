from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.hybrid.sparse_retriever import BM25SparseRetriever
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class SparseRetrieveStep(BaseRetrieverStep):
    def __init__(self, sparse_retriever: BM25SparseRetriever | None = None) -> None:
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        query = HybridRetrieveQuery(
            query=context.active_query,
            knowledge_base_id=context.knowledge_base_id,
            top_k=context.top_k,
            score_threshold=context.score_threshold,
            metadata_filter=context.metadata_filter,
        )
        chunks = self.sparse_retriever.retrieve(query)
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
                    "sparse_scope_applied": True,
                    "sparse_rejected_count": rejected_count,
                }
            )
        context.sparse_chunks = chunks
        context.metadata["sparse_total"] = len(context.sparse_chunks)
        return context
