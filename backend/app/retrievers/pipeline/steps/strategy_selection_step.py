from backend.app.config.settings import settings
from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.strategy import StrategySelector


class StrategySelectionStep(BaseRetrieverStep):
    def __init__(self, selector: StrategySelector | None = None) -> None:
        self.selector = selector or StrategySelector()

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        strategy = self.selector.select(
            routing_result=context.routing_result,
            requested_top_k=context.top_k,
            max_top_k=settings.RETRIEVAL_MAX_TOP_K,
            per_document_top_k=settings.RETRIEVAL_PER_DOCUMENT_TOP_K,
        )
        context.retrieval_strategy = strategy
        context.metadata["retrieval_strategy"] = {
            "strategy": strategy.strategy,
            "route_type": strategy.route_type,
            "document_count": len(strategy.document_ids),
            "per_document_budget": strategy.per_document_budget,
            "global_budget": strategy.global_budget,
            "fallback": strategy.fallback,
            "duration_ms": strategy.metadata.get("duration_ms", 0.0),
        }
        return context
