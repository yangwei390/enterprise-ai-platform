from backend.app.retrievers.metadata_filter import AutoMetadataFilterBuilder
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class MetadataFilterStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        auto_filter_result = AutoMetadataFilterBuilder().build(
            query=context.active_query,
            knowledge_base_id=context.knowledge_base_id,
            metadata_filter=context.metadata_filter,
        )
        context.auto_filter_result = auto_filter_result
        context.metadata["auto_filter_applied"] = auto_filter_result.auto_filter_applied
        context.metadata["candidate_document_ids"] = (
            auto_filter_result.candidate_document_ids
        )
        context.metadata["source_hints"] = auto_filter_result.source_hints
        context.metadata["soft_boost_enabled"] = auto_filter_result.soft_boost_enabled
        context.metadata["auto_filter"] = auto_filter_result.metadata
        return context
