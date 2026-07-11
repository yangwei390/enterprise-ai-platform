from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class LatencyMetric(BaseMetric):
    name = "latency_ms"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        return result.duration_ms, {}


class TokenUsageMetric(BaseMetric):
    name = "token_usage"

    def compute(
        self, case: EvaluationCase, result: EvaluationTargetResult
    ) -> tuple[int | None, dict]:
        value = result.usage.get("total_tokens")
        if isinstance(value, int):
            return value, {"available": True}
        return None, {"available": False}


class EstimatedCostMetric(BaseMetric):
    name = "estimated_cost"

    def compute(
        self, case: EvaluationCase, result: EvaluationTargetResult
    ) -> tuple[float | None, dict]:
        from evaluation.v2.cost import CostCalculator

        provider = str(result.metadata.get("provider") or result.metadata.get("llm_provider") or "")
        model = str(result.output.get("model") or result.metadata.get("model") or "")
        estimate = CostCalculator().estimate(provider, model, result.usage)
        return estimate["estimated_cost"], estimate
