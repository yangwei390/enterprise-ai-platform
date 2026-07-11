import asyncio
import json
from time import perf_counter
from typing import Any

from backend.app.config.settings import settings
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from pydantic import BaseModel, Field


class ReflectionDecision(BaseModel):
    status: str = "continue"
    reason: str = ""
    suggested_action: str = ""
    avoid_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    final_answer: str | None = None


class ReflectionGate:
    def should_reflect(self, state: Any) -> tuple[bool, str | None]:
        if not settings.AGENT_REFLECTION_ENABLED:
            return False, None
        if state.get("explicit_reflect"):
            return True, "explicit_reflect"
        if settings.AGENT_REFLECTION_AFTER_TOOL_FAILURE:
            if any(not result.get("success", True) for result in state.get("tool_results", [])):
                return True, "tool_failed"
        if state.get("same_tool_repeat_count", 0) >= settings.AGENT_REFLECTION_REPEAT_THRESHOLD:
            return True, "repeated_tool"
        recent_observations = state.get("observations", [])
        if recent_observations and not str(recent_observations[-1].get("content", "")).strip():
            return True, "empty_observation"
        return False, None


class ReflectionNodeRunner:
    async def reflect(self, state: Any, reason: str) -> tuple[ReflectionDecision, dict]:
        started_at = perf_counter()
        prompt = {
            "query": state.get("query"),
            "recent_tool_calls": state.get("tool_calls", [])[-5:],
            "recent_observations": state.get("observations", [])[-5:],
            "error_summary": [
                result.get("error")
                for result in state.get("tool_results", [])
                if result.get("error")
            ][-5:],
            "current_step": state.get("step_count", 0),
            "remaining_budget": state.get("budget_remaining", {}),
            "trigger_reason": reason,
        }
        try:
            response = await asyncio.to_thread(
                LLMFactory.get_llm().chat,
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role="system",
                            content=(
                                "你是企业级 Agent Reflection 节点。"
                                "只输出 JSON，不要解释。"
                                "格式：{\"status\":\"continue|replan|final|fail\","
                                "\"reason\":\"...\",\"suggested_action\":\"...\","
                                "\"avoid_tools\":[],\"preferred_tools\":[],"
                                "\"final_answer\":null}"
                            ),
                        ),
                        LLMMessage(
                            role="user",
                            content=json.dumps(prompt, ensure_ascii=False),
                        ),
                    ],
                    temperature=0,
                    metadata={"agent_reflection": True},
                ),
            )
            decision = _parse_reflection(response.answer)
            metadata = {
                "model": response.model,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                "provider_metadata": response.metadata,
            }
            return decision, metadata
        except Exception as exc:
            return (
                ReflectionDecision(
                    status="continue",
                    reason=f"reflection failed: {exc}",
                    suggested_action="continue planning",
                ),
                {
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "error": str(exc),
                },
            )


def _parse_reflection(content: str) -> ReflectionDecision:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return ReflectionDecision(
            status="continue",
            reason="reflection output is not valid json",
        )
    if not isinstance(value, dict):
        return ReflectionDecision(status="continue", reason="reflection output is not object")
    return ReflectionDecision.model_validate(value)
