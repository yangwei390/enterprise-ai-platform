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
        context.sparse_chunks = self.sparse_retriever.retrieve(query)
        context.metadata["sparse_total"] = len(context.sparse_chunks)
        return context
