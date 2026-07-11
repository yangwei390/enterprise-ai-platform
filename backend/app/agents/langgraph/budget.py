from time import perf_counter
from typing import Any

from backend.app.config.settings import settings
from pydantic import BaseModel


class AgentExecutionBudget(BaseModel):
    max_steps: int = settings.AGENT_MAX_STEPS
    max_llm_calls: int = settings.AGENT_MAX_LLM_CALLS
    max_tool_calls: int = settings.AGENT_MAX_TOOL_CALLS
    max_reflections: int = settings.AGENT_MAX_REFLECTIONS
    max_duration_seconds: int = settings.AGENT_MAX_DURATION_SECONDS
    started_at: float = 0

    @classmethod
    def from_settings(cls) -> "AgentExecutionBudget":
        return cls(
            max_steps=settings.AGENT_MAX_STEPS,
            max_llm_calls=settings.AGENT_MAX_LLM_CALLS,
            max_tool_calls=settings.AGENT_MAX_TOOL_CALLS,
            max_reflections=settings.AGENT_MAX_REFLECTIONS,
            max_duration_seconds=settings.AGENT_MAX_DURATION_SECONDS,
            started_at=perf_counter(),
        )


def budget_remaining(state: Any) -> dict:
    budget = state.get("budget") or {}
    return {
        "steps": int(budget.get("max_steps", 0)) - int(state.get("step_count", 0)),
        "llm_calls": int(budget.get("max_llm_calls", 0))
        - int(state.get("llm_call_count", 0)),
        "tool_calls": int(budget.get("max_tool_calls", 0))
        - int(state.get("tool_call_count", 0)),
        "reflections": int(budget.get("max_reflections", 0))
        - int(state.get("reflection_count", 0)),
    }


def check_budget(state: Any, *, before: str) -> str | None:
    budget = state.get("budget") or {}
    if int(state.get("step_count", 0)) >= int(budget.get("max_steps", 0)):
        return "max_steps_exceeded"
    if before == "llm" and int(state.get("llm_call_count", 0)) >= int(
        budget.get("max_llm_calls", 0)
    ):
        return "max_llm_calls_exceeded"
    if before == "tool" and int(state.get("tool_call_count", 0)) >= int(
        budget.get("max_tool_calls", 0)
    ):
        return "max_tool_calls_exceeded"
    if before == "reflection" and int(state.get("reflection_count", 0)) >= int(
        budget.get("max_reflections", 0)
    ):
        return "max_reflections_exceeded"
    started_at = float(budget.get("started_at") or perf_counter())
    if perf_counter() - started_at >= float(budget.get("max_duration_seconds", 0)):
        return "max_duration_exceeded"
    return None


def mark_budget_exceeded(state: Any, reason: str) -> None:
    state["loop_status"] = "terminated"
    state["termination_reason"] = reason
    state["current_action"] = "final"
    state.setdefault("metadata", {}).setdefault("agent_loop", {})[
        "budget_exceeded"
    ] = True
