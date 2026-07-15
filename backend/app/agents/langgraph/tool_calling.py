import asyncio
import json
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any
from uuid import uuid4

from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.tool_scope import (
    build_tool_scope,
    filter_descriptors_by_scope,
    tool_scope_trace,
)
from backend.app.config.settings import settings
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.tools import get_tool_registry
from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    index: int = 0
    metadata: dict = Field(default_factory=dict)


class AgentDecision(BaseModel):
    action: str
    content: str | None = None
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class BaseAgentPlannerStrategy(ABC):
    name: str

    @abstractmethod
    async def adecide(self, state: Any) -> AgentDecision:
        raise NotImplementedError


class NativeToolCallingStrategy(BaseAgentPlannerStrategy):
    name = "native_tool_calling"

    async def adecide(self, state: Any) -> AgentDecision:
        registry = get_tool_registry()
        scope = build_tool_scope(state.get("metadata", {}) if isinstance(state, dict) else {})
        descriptors = filter_descriptors_by_scope(
            registry.list_descriptors(enabled_only=True),
            scope,
        )
        llm = LLMFactory.get_llm()
        if not getattr(llm, "supports_tool_calling", False):
            if settings.AGENT_PLANNER_FALLBACK_ENABLED:
                decision = await JsonPlanStrategy().adecide(state)
                decision.metadata.update(
                    {
                        "requested_strategy": self.name,
                        "actual_strategy": "json_plan",
                        "fallback_used": True,
                        "fallback_reason": "provider does not support native tool calling",
                    }
                )
                return decision
            return AgentDecision(
                action="final",
                content="当前模型不支持原生工具调用，且未启用 Planner fallback。",
                metadata={"native_supported": False},
            )

        started_at = perf_counter()
        model_config = _agent_model_config(state)
        response = await asyncio.to_thread(
            llm.chat,
            LLMRequest(
                messages=[LLMMessage.model_validate(message) for message in state["messages"]],
                model=model_config.get("model"),
                temperature=float(model_config.get("temperature", 0)),
                tools=[descriptor.to_llm_schema() for descriptor in descriptors],
                tool_choice="auto",
                parallel_tool_calls=True,
                metadata={
                    "agent_planner_strategy": self.name,
                    "agent_id": state.get("metadata", {}).get("agent_id"),
                    "agent_model_config_keys": sorted(model_config),
                },
            ),
        )
        allowed_tools = {descriptor.name for descriptor in descriptors}
        tool_calls: list[AgentToolCall] = []
        rejected_tools: list[str] = []
        for index, tool_call in enumerate(response.tool_calls):
            if tool_call.name not in allowed_tools:
                rejected_tools.append(tool_call.name)
                continue
            tool_calls.append(
                AgentToolCall(
                    id=tool_call.id or f"tool_call_{uuid4().hex}",
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    index=index,
                    metadata={"finish_reason": response.finish_reason},
                )
            )

        metadata = {
            "requested_strategy": self.name,
            "actual_strategy": self.name,
            "fallback_used": False,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "finish_reason": response.finish_reason,
            "provider_metadata": response.metadata,
            "usage": response.usage,
            "tool_registry": {
                "available_tool_count": len(descriptors),
                "available_tools": sorted(allowed_tools),
                "selected_tools": [tool.tool_name for tool in tool_calls],
                "rejected_tools": rejected_tools,
                "registry_version": registry.version,
                "refreshed_at": registry.last_refresh,
            },
            "tool_scope": tool_scope_trace(
                scope=scope,
                visible_tools=sorted(allowed_tools),
                selected_tools=[tool.tool_name for tool in tool_calls],
                blocked_tools=rejected_tools,
                permission_result="blocked" if rejected_tools else "allowed",
                permission_reason="tool_not_allowed" if rejected_tools else None,
            ),
        }
        if rejected_tools and not tool_calls:
            return AgentDecision(
                action="reflect",
                content=response.answer,
                metadata={**metadata, "error": "unknown_or_disabled_tool"},
            )
        if tool_calls:
            return AgentDecision(
                action="tool_calls",
                content=response.answer,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        return AgentDecision(action="final", content=response.answer, metadata=metadata)


class JsonPlanStrategy(BaseAgentPlannerStrategy):
    name = "json_plan"

    async def adecide(self, state: Any) -> AgentDecision:
        scope = build_tool_scope(state.get("metadata", {}) if isinstance(state, dict) else {})
        plan = await LLMPlanner().aplan(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id"),
            conversation_id=state.get("conversation_id"),
            memory_context=state.get("memory_context"),
            tool_allowlist=list(scope.allowed_tools) if not scope.unrestricted else None,
            tool_scope_source=scope.source,
        )
        allowed = set(scope.allowed_tools)
        blocked_tools: list[str] = []
        tool_calls = [
            AgentToolCall(
                id=f"json_plan_{index}_{uuid4().hex}",
                tool_name=step.tool,
                arguments=step.args,
                index=index,
            )
            for index, step in enumerate(plan.steps)
            if scope.unrestricted or step.tool in allowed
        ]
        if not scope.unrestricted:
            blocked_tools = [step.tool for step in plan.steps if step.tool not in allowed]
        selected_tools = [tool.tool_name for tool in tool_calls]
        tool_scope_metadata = tool_scope_trace(
            scope=scope,
            visible_tools=plan.metadata.get("tool_registry", {}).get("available_tools", []),
            selected_tools=selected_tools,
            blocked_tools=blocked_tools,
            permission_result="blocked" if blocked_tools else "allowed",
            permission_reason="tool_not_allowed" if blocked_tools else None,
        )
        return AgentDecision(
            action="tool_calls" if tool_calls else "final",
            content=None if tool_calls else "当前 Agent V2 未选择工具，直接回答能力待增强。",
            tool_calls=tool_calls,
            metadata={
                **plan.metadata,
                "requested_strategy": _planner_strategy_name(state),
                "actual_strategy": self.name,
                "fallback_used": _planner_strategy_name(state) != self.name,
                "tool_scope": tool_scope_metadata,
            },
        )


def get_planner_strategy(state: Any | None = None) -> BaseAgentPlannerStrategy:
    if _planner_strategy_name(state) == "json_plan":
        return JsonPlanStrategy()
    return NativeToolCallingStrategy()


def _planner_strategy_name(state: Any | None) -> str:
    if isinstance(state, dict):
        value = state.get("metadata", {}).get("planner_strategy")
        if isinstance(value, str) and value:
            return value
    return settings.AGENT_PLANNER_STRATEGY


def _agent_model_config(state: Any) -> dict:
    if isinstance(state, dict):
        config = (
            state.get("metadata", {})
            .get("agent_definition", {})
            .get("model_config", {})
        )
        if isinstance(config, dict):
            return config
    return {}


def assistant_tool_calls_payload(tool_calls: list[AgentToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.tool_name,
                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
            },
        }
        for tool_call in tool_calls
    ]
