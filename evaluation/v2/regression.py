from typing import Any


class RegressionComparator:
    HIGHER_IS_BETTER = {
        "retriever_hit",
        "chunk_recall",
        "keyword_coverage",
        "mrr",
        "hit_rate_at_k",
        "context_precision_proxy",
        "tool_selection_accuracy",
        "tool_call_success_rate",
    }
    LOWER_IS_BETTER = {"latency_ms", "estimated_cost", "retry_count", "step_count"}

    def compare(self, current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        current_cases = {case["case_id"]: case for case in current.get("cases", [])}
        baseline_cases = {case["case_id"]: case for case in baseline.get("cases", [])}
        comparisons = []
        for case_id, current_case in current_cases.items():
            baseline_case = baseline_cases.get(case_id)
            if not baseline_case:
                continue
            baseline_metrics = _metric_map(baseline_case)
            for metric in current_case.get("metrics", []):
                name = metric.get("name")
                if name not in baseline_metrics:
                    continue
                current_value = metric.get("value")
                baseline_value = baseline_metrics[name].get("value")
                if not isinstance(current_value, int | float) or not isinstance(
                    baseline_value, int | float
                ):
                    continue
                delta = current_value - baseline_value
                status = self._status(name, current_value, baseline_value)
                comparisons.append(
                    {
                        "case_id": case_id,
                        "metric": name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "delta": delta,
                        "status": status,
                    }
                )
        return {
            "comparisons": comparisons,
            "improved_metrics": sum(1 for item in comparisons if item["status"] == "improved"),
            "regressed_metrics": sum(1 for item in comparisons if item["status"] == "regressed"),
            "unchanged_metrics": sum(1 for item in comparisons if item["status"] == "unchanged"),
            "regressed_cases": sorted(
                {item["case_id"] for item in comparisons if item["status"] == "regressed"}
            ),
        }

    def _status(self, name: str, current: float, baseline: float) -> str:
        if current == baseline:
            return "unchanged"
        if name in self.LOWER_IS_BETTER:
            return "improved" if current < baseline else "regressed"
        return "improved" if current > baseline else "regressed"


def _metric_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {metric.get("name"): metric for metric in case.get("metrics", [])}
