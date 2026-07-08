from backend.app.context_compression import (
    CompressionInput,
    ContextCompressorFactory,
    get_context_compression_config,
)
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class ContextCompressionStep(BaseRetrieverStep):
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        config = get_context_compression_config()
        metadata = {
            "enabled": config.enabled,
            "provider": config.provider,
            "original_chunk_count": len(context.context_chunks),
            "compressed_chunk_count": len(context.context_chunks),
            "original_chars": len(context.context_text),
            "compressed_chars": len(context.context_text),
            "skipped_chunk_count": 0,
            "max_chars": config.max_chars,
            "max_chunk_chars": config.max_chunk_chars,
            "failed": False,
            "error": None,
        }

        if not config.enabled:
            context.metadata["context_compression"] = metadata
            return context

        try:
            compressor = ContextCompressorFactory.get_compressor(config.provider)
            result = compressor.compress(
                CompressionInput(
                    query=context.active_query,
                    chunks=context.context_chunks,
                    max_chars=config.max_chars,
                    metadata={
                        "context_text": context.context_text,
                    },
                )
            )
            context.context_chunks = result.compressed_chunks
            context.context_text = str(result.metadata.get("context_text") or "")
            context.metadata["context_compression"] = {
                **metadata,
                "original_chunk_count": result.original_chunk_count,
                "compressed_chunk_count": result.compressed_chunk_count,
                "original_chars": result.original_chars,
                "compressed_chars": result.compressed_chars,
                "skipped_chunk_count": result.skipped_chunk_count,
                **result.metadata,
                "failed": False,
                "error": None,
            }
            return context
        except Exception as exc:
            context.add_error("ContextCompressionStep", exc)
            metadata["failed"] = True
            metadata["error"] = str(exc)
            context.metadata["context_compression"] = metadata
            if not config.fail_open:
                raise
            return context
