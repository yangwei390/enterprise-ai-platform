import json
from pathlib import Path
from typing import Any

from backend.app.config.settings import settings


class CostCalculator:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path or settings.EVALUATION_COST_CONFIG_PATH)
        self.prices = self._load_prices()

    def estimate(self, provider: str, model: str, usage: dict[str, Any]) -> dict[str, Any]:
        price = self.prices.get((provider, model))
        input_tokens = _token_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _token_value(usage, "output_tokens", "completion_tokens")
        if price is None or input_tokens is None or output_tokens is None:
            return {"estimated_cost": None, "currency": None, "available": False}
        cost = (
            input_tokens / 1_000_000 * price["input_cost_per_1m_tokens"]
            + output_tokens / 1_000_000 * price["output_cost_per_1m_tokens"]
        )
        return {"estimated_cost": cost, "currency": price["currency"], "available": True}

    def _load_prices(self) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.config_path.exists():
            return {}
        text = self.config_path.read_text(encoding="utf-8")
        if self.config_path.suffix == ".json":
            raw = json.loads(text)
        else:
            try:
                import yaml  # type: ignore[import-untyped]

                raw = yaml.safe_load(text)
            except ModuleNotFoundError:
                from evaluation.v2.suite import _load_limited_yaml

                raw = _load_limited_yaml(text)
        items = raw.get("providers", []) if isinstance(raw, dict) else []
        return {
            (str(item["provider"]), str(item["model"])): item
            for item in items
            if isinstance(item, dict) and "provider" in item and "model" in item
        }


def _token_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None
