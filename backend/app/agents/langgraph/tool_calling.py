import asyncio
import json
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any
from uuid import uuid4

from backend.app.agents.langgraph.planner import LLMPlanner
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
        descriptors = registry.list_descriptors(enabled_only=True)
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
        response = await asyncio.to_thread(
            llm.chat,
            LLMRequest(
                messages=[LLMMessage.model_validate(message) for message in state["messages"]],
                tools=[descriptor.to_llm_schema() for descriptor in descriptors],
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0,
                metadata={"agent_planner_strategy": self.name},
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
        plan = await LLMPlanner().aplan(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id"),
            conversation_id=state.get("conversation_id"),
            memory_context=state.get("memory_context"),
        )
        tool_calls = [
            AgentToolCall(
                id=f"json_plan_{index}_{uuid4().hex}",
                tool_name=step.tool,
                arguments=step.args,
                index=index,
            )
            for index, step in enumerate(plan.steps)
        ]
        return AgentDecision(
            action="tool_calls" if tool_calls else "final",
            content=None if tool_calls else "当前 Agent V2 未选择工具，直接回答能力待增强。",
            tool_calls=tool_calls,
            metadata={
                **plan.metadata,
                "requested_strategy": settings.AGENT_PLANNER_STRATEGY,
                "actual_strategy": self.name,
                "fallback_used": settings.AGENT_PLANNER_STRATEGY != self.name,
            },
        )


def get_planner_strategy() -> BaseAgentPlannerStrategy:
    if settings.AGENT_PLANNER_STRATEGY == "json_plan":
        return JsonPlanStrategy()
    return NativeToolCallingStrategy()


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
