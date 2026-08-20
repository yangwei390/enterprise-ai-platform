from __future__ import annotations

from typing import Any, Literal

from backend.app.config import settings
from pydantic import BaseModel, ConfigDict, Field

StageMode = Literal["rule_only", "llm_only", "hybrid"]


class StageExecutionDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: StageMode
    source: Literal["rule", "llm", "shared_llm", "safety", "none"]
    executed: bool = False
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    shared_call_id: str | None = None


def query_rewrite_mode() -> StageMode:
    return settings.CUSTOMER_SERVICE_QUERY_REWRITE_MODE


def intent_routing_mode() -> StageMode:
    return settings.CUSTOMER_SERVICE_INTENT_ROUTING_MODE


def slot_extraction_mode() -> StageMode:
    return settings.CUSTOMER_SERVICE_SLOT_EXTRACTION_MODE


def tool_selection_mode() -> StageMode:
    return settings.CUSTOMER_SERVICE_TOOL_SELECTION_MODE


def reference_interpretation_mode() -> StageMode:
    return settings.CUSTOMER_SERVICE_REFERENCE_INTERPRETATION_MODE


def all_stage_modes() -> dict[str, StageMode]:
    return {
        "query_rewrite": query_rewrite_mode(),
        "intent_routing": intent_routing_mode(),
        "slot_extraction": slot_extraction_mode(),
        "capability_selection": tool_selection_mode(),
        "reference_interpretation": reference_interpretation_mode(),
    }
