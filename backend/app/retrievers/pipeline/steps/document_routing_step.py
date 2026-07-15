from time import perf_counter

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.retrievers.document_router import DocumentRouter, RoutingResult
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class DocumentRoutingStep(BaseRetrieverStep):
    def __init__(self, router: DocumentRouter | None = None) -> None:
        self.router = router or DocumentRouter()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        if not settings.DOCUMENT_ROUTER_ENABLED:
            result = RoutingResult(
                route_type="KNOWLEDGE_BASE",
                reason="disabled",
                confidence=0.0,
            )
            context.routing_result = result
            self._write_metadata(context, result, duration_ms=0.0)
            return context

        started = perf_counter()
        try:
            result = self.router.route(
                understanding=context.query_understanding,
                rewrite_result=context.query_rewrite_result,
                metadata_filter_result=context.auto_filter_result,
                max_candidates=settings.DOCUMENT_ROUTER_MAX_CANDIDATES,
            )
            context.routing_result = result
            duration_ms = result.metadata.get(
                "duration_ms",
                round((perf_counter() - started) * 1000, 2),
            )
            self._write_metadata(context, result, duration_ms=duration_ms)
            return context
        except Exception as exc:
            logger.exception("Document routing failed")
            if not settings.DOCUMENT_ROUTER_FAIL_OPEN:
                raise
            result = RoutingResult(
                route_type="KNOWLEDGE_BASE",
                reason="fail_open",
                confidence=0.0,
                metadata={"error": str(exc)},
            )
            context.routing_result = result
            self._write_metadata(
                context,
                result,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                failed=True,
                error=str(exc),
            )
            return context

    def _write_metadata(
        self,
        context: RetrieverPipelineContext,
        result: RoutingResult,
        *,
        duration_ms: float,
        failed: bool = False,
        error: str | None = None,
    ) -> None:
        context.metadata["document_routing"] = {
            "route_type": result.route_type,
            "confidence": result.confidence,
            "reason": result.reason,
            "candidate_count": len(result.candidate_document_ids),
            "selected_count": len(result.target_document_ids),
            "candidate_document_ids": result.candidate_document_ids,
            "target_document_ids": result.target_document_ids,
            "duration_ms": duration_ms,
            "failed": failed,
            "error": error,
        }
