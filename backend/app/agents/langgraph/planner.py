import asyncio
import json
from typing import Any

from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.tools import get_tool_registry
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class LLMPlanner:
    async def aplan(
        self,
        *,
        query: str,
        knowledge_base_id: int | None = None,
        conversation_id: int | None = None,
        memory_context: str | None = None,
        tool_allowlist: list[str] | None = None,
        tool_scope_source: str | None = None,
    ) -> AgentPlan:
        return await asyncio.to_thread(
            self.plan,
            query=query,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
            memory_context=memory_context,
            tool_allowlist=tool_allowlist,
            tool_scope_source=tool_scope_source,
        )

    def plan(
        self,
        *,
        query: str,
        knowledge_base_id: int | None = None,
        conversation_id: int | None = None,
        memory_context: str | None = None,
        tool_allowlist: list[str] | None = None,
        tool_scope_source: str | None = None,
    ) -> AgentPlan:
        registry = get_tool_registry()
        allowed_names = set(tool_allowlist) if tool_allowlist is not None else None
        tool_descriptors = [
            descriptor
            for descriptor in registry.list_descriptors(enabled_only=True)
            if allowed_names is None or descriptor.name in allowed_names
        ]
        tool_schemas = [descriptor.to_llm_schema() for descriptor in tool_descriptors]
        if not tool_descriptors:
            return AgentPlan(
                steps=[],
                metadata={
                    "tool_registry": {
                        "available_tool_count": 0,
                        "available_tools": [],
                        "selected_tools": [],
                        "rejected_tools": [],
                        "registry_version": registry.version,
                        "refreshed_at": registry.last_refresh,
                        "tool_scope_source": tool_scope_source,
                    }
                },
            )
        response = LLMFactory.get_llm().chat(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "你是企业级 Agent Planner。"
                            "请根据用户问题输出 JSON Plan。"
                            "只能输出 JSON，不要解释。"
                            "格式：{\"steps\":[{\"tool\":\"工具名\",\"args\":{}}]}。"
                            "如果不需要工具，输出：{\"steps\":[]}。"
                            "只能选择 available_tools 中存在的工具。"
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "query": query,
                                "knowledge_base_id": knowledge_base_id,
                                "conversation_id": conversation_id,
                                "memory_context": memory_context,
                                "available_tools": tool_schemas,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                metadata={"agent_runtime": "langgraph_v2", "planner": "llm"},
            )
        )
        plan_data = self._parse_plan(response.answer)
        return self._normalize_plan(
            plan_data=plan_data,
            registry_version=registry.version,
            refreshed_at=registry.last_refresh,
            tool_descriptors={descriptor.name: descriptor for descriptor in tool_descriptors},
            query=query,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
            memory_context=memory_context,
            tool_scope_source=tool_scope_source,
        )

    def _parse_plan(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"steps": [], "metadata": {"parse_failed": True, "raw": content}}
        if not isinstance(parsed, dict):
            return {"steps": [], "metadata": {"parse_failed": True, "raw": content}}
        return parsed

    def _normalize_plan(
        self,
        *,
        plan_data: dict[str, Any],
        registry_version: int | None = None,
        refreshed_at: dict | None = None,
        tool_descriptors: dict | None = None,
        query: str,
        knowledge_base_id: int | None,
        conversation_id: int | None,
        memory_context: str | None,
        tool_scope_source: str | None = None,
    ) -> AgentPlan:
        registry = get_tool_registry()
        descriptors = tool_descriptors or {
            descriptor.name: descriptor
            for descriptor in registry.list_descriptors(enabled_only=True)
        }
        allowed_tools = set(descriptors)
        raw_steps = plan_data.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: list[PlanStep] = []
        rejected_tools: list[str] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            tool_name = raw_step.get("tool")
            if not isinstance(tool_name, str) or tool_name not in allowed_tools:
                if isinstance(tool_name, str):
                    rejected_tools.append(tool_name)
                continue
            args = raw_step.get("args")
            if not isinstance(args, dict):
                args = {}
            args = self._fill_default_args(
                args=args,
                descriptor=descriptors[tool_name],
                query=query,
                knowledge_base_id=knowledge_base_id,
                conversation_id=conversation_id,
                memory_context=memory_context,
            )
            depends_on = raw_step.get("depends_on", [])
            steps.append(
                PlanStep(
                    tool=tool_name,
                    args=args,
                    depends_on=depends_on if isinstance(depends_on, list) else [],
                )
            )

        metadata = plan_data.get("metadata")
        selected_tools = [step.tool for step in steps]
        tool_registry_metadata = {
            "available_tool_count": len(allowed_tools),
            "available_tools": sorted(allowed_tools),
            "selected_tools": selected_tools,
            "rejected_tools": rejected_tools,
            "registry_version": registry_version
            if registry_version is not None
            else registry.version,
            "refreshed_at": refreshed_at if refreshed_at is not None else registry.last_refresh,
            "tool_scope_source": tool_scope_source,
        }
        merged_metadata = metadata if isinstance(metadata, dict) else {}
        return AgentPlan(
            steps=steps,
            metadata={
                **merged_metadata,
                "tool_registry": tool_registry_metadata,
            },
        )

    def _fill_default_args(
        self,
        *,
        args: dict,
        descriptor,
        query: str,
        knowledge_base_id: int | None,
        conversation_id: int | None,
        memory_context: str | None,
    ) -> dict:
        properties = descriptor.input_schema.get("properties", {})
        if not isinstance(properties, dict):
            return args

        defaults = {
            "query": query,
            "knowledge_base_id": knowledge_base_id,
            "conversation_id": conversation_id,
            "memory_context": memory_context,
        }
        filled_args = dict(args)
        for key, value in defaults.items():
            if key in properties and filled_args.get(key) is None:
                filled_args[key] = value
        return filled_args
