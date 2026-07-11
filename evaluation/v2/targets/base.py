from abc import ABC, abstractmethod
from time import perf_counter

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult


class BaseEvaluationTarget(ABC):
    name: str

    @abstractmethod
    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        raise NotImplementedError


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
