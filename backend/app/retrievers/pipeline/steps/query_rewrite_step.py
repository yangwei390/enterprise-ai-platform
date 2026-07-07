from backend.app.query import SimpleQueryRewriter
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class QueryRewriteStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        rewrite_result = SimpleQueryRewriter().rewrite(context.query)
        context.original_query = rewrite_result.original_query
        context.rewritten_query = rewrite_result.rewritten_query
        context.metadata["query_rewrite"] = rewrite_result.model_dump()
        return context
