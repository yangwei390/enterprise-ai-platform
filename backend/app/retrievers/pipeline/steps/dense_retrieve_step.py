from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class DenseRetrieveStep(BaseRetrieverStep):
    def __init__(self, dense_retriever: DenseRetriever | None = None) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        query = HybridRetrieveQuery(
            query=context.active_query,
            knowledge_base_id=context.knowledge_base_id,
            top_k=context.top_k,
            score_threshold=context.score_threshold,
            metadata_filter=context.metadata_filter,
        )
        context.dense_chunks = self.dense_retriever.retrieve(query)
        context.metadata["dense_total"] = len(context.dense_chunks)
        return context
