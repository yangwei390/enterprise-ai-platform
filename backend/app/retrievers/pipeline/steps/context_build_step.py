from backend.app.config.settings import settings
from backend.app.context import (
    BasicContextBuilder,
    ContextBuildRequest,
)
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class ContextBuildStep(BaseRetrieverStep):
    def __init__(self, builder: BasicContextBuilder | None = None) -> None:
        self.builder = builder or BasicContextBuilder()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        request = ContextBuildRequest(
            query=context.active_query,
            chunks=context.reranked_chunks,
            max_context_chars=settings.CONTEXT_MAX_CHARS,
            route_type=(
                context.routing_result.route_type
                if context.routing_result is not None
                else None
            ),
            strategy=(
                context.retrieval_strategy.strategy
                if context.retrieval_strategy is not None
                else None
            ),
            target_document_ids=(
                context.routing_result.target_document_ids
                if context.routing_result is not None
                else []
            ),
            max_chunks=settings.CONTEXT_MAX_CHUNKS,
            max_chars_per_document=settings.CONTEXT_MAX_CHARS_PER_DOCUMENT,
            multi_document_diversity_enabled=(
                settings.CONTEXT_MULTI_DOCUMENT_DIVERSITY_ENABLED
            ),
            multi_document_max_per_document=(
                settings.CONTEXT_MULTI_DOCUMENT_MAX_PER_DOCUMENT
            ),
            multi_document_min_documents=(
                settings.CONTEXT_MULTI_DOCUMENT_MIN_DOCUMENTS
            ),
        )
        context_result = self.builder.build(request)
        context.context_chunks = context_result.chunks
        context.context_text = context_result.context_text
        context.metadata["context_builder"] = context_result.metadata
        return context
