from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.query import SimpleQueryRewriter
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class QueryRewriteStep(BaseRetrieverStep):
    def __init__(
        self,
        rewriter: SimpleQueryRewriter | None = None,
    ) -> None:
        self.rewriter = rewriter or SimpleQueryRewriter()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        if not settings.QUERY_REWRITE_ENABLED:
            context.original_query = context.query
            context.rewritten_query = context.query
            context.metadata["query_rewrite"] = {
                "enabled": False,
                "rewrite_type": "NONE",
                "changed": False,
                "reason": "disabled",
                "before": context.query,
                "after": context.query,
                "duration_ms": 0.0,
                "failed": False,
                "error": None,
            }
            return context

        try:
            rewrite_result = self.rewriter.rewrite(
                context.query,
                understanding=context.query_understanding,
                max_length=settings.QUERY_REWRITE_MAX_LENGTH,
            )
        except Exception as exc:
            logger.exception("Query rewrite failed")
            if not settings.QUERY_REWRITE_FAIL_OPEN:
                raise
            context.original_query = context.query
            context.rewritten_query = context.query
            context.metadata["query_rewrite"] = {
                "enabled": True,
                "rewrite_type": "NONE",
                "changed": False,
                "reason": "fail_open",
                "before": context.query,
                "after": context.query,
                "duration_ms": 0.0,
                "failed": True,
                "error": str(exc),
            }
            return context

        context.original_query = rewrite_result.original_query
        context.rewritten_query = rewrite_result.rewritten_query
        context.query_rewrite_result = rewrite_result
        context.metadata["query_rewrite"] = {
            "enabled": True,
            "rewrite_type": rewrite_result.rewrite_type,
            "changed": rewrite_result.rewrite_changed,
            "reason": rewrite_result.rewrite_reason,
            "before": rewrite_result.original_query,
            "after": rewrite_result.rewritten_query,
            "duration_ms": rewrite_result.duration_ms,
            "failed": False,
            "error": None,
            "metadata": rewrite_result.metadata,
        }
        return context
