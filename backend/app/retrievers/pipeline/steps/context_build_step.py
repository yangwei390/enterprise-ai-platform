from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.context.compression import SimpleContextCompressor
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
        compression_result = SimpleContextCompressor().compress(
            context_text=context_result.context_text,
            chunks=context_result.chunks,
        )
        context.context_chunks = compression_result.chunks
        context.context_text = compression_result.context_text
        context.metadata["context_builder"] = context_result.metadata
        context.metadata["context_compression"] = {
            "original_chars": compression_result.original_chars,
            "compressed_chars": compression_result.compressed_chars,
            "compression_applied": compression_result.compression_applied,
            **compression_result.metadata,
        }
        return context
