from typing import Any


class ThresholdEvaluator:
    def evaluate(self, value: Any, threshold: Any) -> bool:
        if threshold is None:
            return True
        if isinstance(threshold, dict):
            if "min" in threshold and (value is None or value < threshold["min"]):
                return False
            if "max" in threshold and (value is None or value > threshold["max"]):
                return False
            return True
        return value == threshold
