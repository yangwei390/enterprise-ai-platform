from backend.app.retrievers.pipeline.base import BaseRetrieverStep
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps import (
    ContextBuildStep,
    ContextCompressionStep,
    DenseRetrieveStep,
    FusionStep,
    MetadataFilterStep,
    MMRStep,
    NeighborExpansionStep,
    QueryRewriteStep,
    RerankStep,
    RetrievalPlanningStep,
    SoftBoostStep,
    SparseRetrieveStep,
)


class RetrieverPipeline:
    def __init__(self, steps: list[BaseRetrieverStep] | None = None) -> None:
        self.steps = steps or [
            QueryRewriteStep(),
            MetadataFilterStep(),
            RetrievalPlanningStep(),
            DenseRetrieveStep(),
            SparseRetrieveStep(),
            FusionStep(),
            SoftBoostStep(),
            RerankStep(),
            MMRStep(),
            NeighborExpansionStep(),
            ContextBuildStep(),
            ContextCompressionStep(),
        ]

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        for step in self.steps:
            try:
                context = step.run(context)
            except Exception as exc:
                context.add_error(step.__class__.__name__, exc)
                raise
        return context
