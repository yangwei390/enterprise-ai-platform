from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.query_understanding import (
    FastQueryAnalyzer,
    QueryUnderstandingTrace,
)


class QueryUnderstandingStep(BaseRetrieverStep):
    def __init__(self, analyzer: FastQueryAnalyzer | None = None) -> None:
        self.analyzer = analyzer or FastQueryAnalyzer()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        if not settings.QUERY_UNDERSTANDING_ENABLED:
            context.metadata["query_understanding"] = QueryUnderstandingTrace(
                enabled=False,
            ).model_dump()
            return context

        if context.query_understanding is not None:
            result = context.query_understanding
            context.metadata["query_understanding"] = QueryUnderstandingTrace(
                enabled=True,
                intent=result.intent,
                confidence=result.confidence,
                keywords=result.keywords,
                entities=result.entities,
                document_hints=result.document_hints,
                structure_hints=result.structure_hints,
                temporal_constraints=result.temporal_constraints,
                comparison_targets=result.comparison_targets,
                negative_constraints=result.negative_constraints,
                analyzer_source=result.analyzer_source,
                duration_ms=result.duration_ms,
                failed=False,
            ).model_dump()
            return context

        try:
            result = self.analyzer.analyze(context.active_query)
            context.query_understanding = result
            context.metadata["query_understanding"] = QueryUnderstandingTrace(
                enabled=True,
                intent=result.intent,
                confidence=result.confidence,
                keywords=result.keywords,
                entities=result.entities,
                document_hints=result.document_hints,
                structure_hints=result.structure_hints,
                temporal_constraints=result.temporal_constraints,
                comparison_targets=result.comparison_targets,
                negative_constraints=result.negative_constraints,
                analyzer_source=result.analyzer_source,
                duration_ms=result.duration_ms,
                failed=False,
            ).model_dump()
            return context
        except Exception as exc:
            logger.exception("Query understanding failed")
            context.metadata["query_understanding"] = QueryUnderstandingTrace(
                enabled=True,
                failed=True,
                error=str(exc),
            ).model_dump()
            if not settings.QUERY_UNDERSTANDING_FAIL_OPEN:
                raise
            return context
