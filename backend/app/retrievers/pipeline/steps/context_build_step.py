from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class ContextBuildStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        context_result = ContextBuilderFactory.get_builder().build(
            ContextBuildRequest(
                query=context.active_query,
                chunks=context.reranked_chunks,
            )
        )
        context.context_chunks = context_result.chunks
        context.context_text = context_result.context_text
        context.metadata["context_builder"] = context_result.metadata
        return context
