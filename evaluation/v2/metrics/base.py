from abc import ABC, abstractmethod
from time import perf_counter

from evaluation.v2.schemas import (
    EvaluationCase,
    EvaluationContext,
    EvaluationTargetResult,
    MetricResult,
)
from evaluation.v2.threshold import ThresholdEvaluator


class BaseMetric(ABC):
    name: str

    async def evaluate(
        self,
        case: EvaluationCase,
        result: EvaluationTargetResult,
        context: EvaluationContext,
    ) -> MetricResult:
        started_at = perf_counter()
        try:
            value, details = self.compute(case, result)
            threshold = case.thresholds.get(self.name)
            passed = ThresholdEvaluator().evaluate(value, threshold)
            return MetricResult(
                name=self.name,
                value=value,
                passed=passed,
                threshold=threshold,
                details=details,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                value=None,
                passed=False,
                error=str(exc),
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )

    @abstractmethod
    def compute(
        self,
        case: EvaluationCase,
        result: EvaluationTargetResult,
    ) -> tuple[object, dict]:
        raise NotImplementedError
