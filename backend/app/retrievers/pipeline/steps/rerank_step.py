from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class RerankStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        rerank_result = RerankerFactory.get_reranker().rerank(
            RerankQuery(
                query=context.active_query,
                chunks=context.fused_chunks,
                top_k=context.top_k,
            )
        )
        context.reranked_chunks = rerank_result.chunks
        context.metadata["reranker"] = rerank_result.metadata
        context.metadata["reranked_total"] = rerank_result.total
        return context
