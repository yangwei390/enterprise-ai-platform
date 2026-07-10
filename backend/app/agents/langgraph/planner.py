import json
from typing import Any

from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.tools import get_tool_registry
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class AgentPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class LLMPlanner:
    def plan(
        self,
        *,
        query: str,
        knowledge_base_id: int | None = None,
        conversation_id: int | None = None,
        memory_context: str | None = None,
    ) -> AgentPlan:
        tool_definitions = [
            definition.model_dump()
            for definition in get_tool_registry().get_tool_definitions()
        ]
        response = LLMFactory.get_llm().chat(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "你是企业级 Agent Planner。"
                            "请根据用户问题输出 JSON Plan。"
                            "只能输出 JSON，不要解释。"
                            "格式：{\"steps\":[{\"tool\":\"knowledge_search\",\"args\":{}}]}。"
                            "如果不需要工具，输出：{\"steps\":[]}。"
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
                                "available_tools": tool_definitions,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                metadata={"agent_runtime": "langgraph", "planner": "llm"},
            )
        )
        plan_data = self._parse_plan(response.answer)
        return self._normalize_plan(
            plan_data=plan_data,
            query=query,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
            memory_context=memory_context,
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
        query: str,
        knowledge_base_id: int | None,
        conversation_id: int | None,
        memory_context: str | None,
    ) -> AgentPlan:
        allowed_tools = {
            definition.name
            for definition in get_tool_registry().get_tool_definitions()
        }
        raw_steps = plan_data.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: list[PlanStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            tool_name = raw_step.get("tool")
            if not isinstance(tool_name, str) or tool_name not in allowed_tools:
                continue
            args = raw_step.get("args")
            if not isinstance(args, dict):
                args = {}
            if tool_name == "knowledge_search":
                args = {
                    "query": args.get("query") or query,
                    "knowledge_base_id": args.get(
                        "knowledge_base_id",
                        knowledge_base_id,
                    ),
                    "conversation_id": args.get("conversation_id", conversation_id),
                    "memory_context": args.get("memory_context", memory_context),
                }
            steps.append(PlanStep(tool=tool_name, args=args))

        metadata = plan_data.get("metadata")
        return AgentPlan(
            steps=steps,
            metadata=metadata if isinstance(metadata, dict) else {},
        )
