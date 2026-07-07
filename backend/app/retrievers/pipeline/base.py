from abc import ABC, abstractmethod

from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


class BaseRetrieverStep(ABC):
    @abstractmethod
    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        raise NotImplementedError
