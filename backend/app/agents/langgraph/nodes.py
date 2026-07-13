import asyncio
import hashlib
import json
from inspect import isawaitable
from time import perf_counter
from typing import Any, cast

from backend.app.agents.final_answer import (
    build_final_answer_request,
    collect_streaming_answer,
)
from backend.app.agents.langgraph.budget import (
    AgentExecutionBudget,
    budget_remaining,
    check_budget,
    mark_budget_exceeded,
)
from backend.app.agents.langgraph.reflection import ReflectionGate, ReflectionNodeRunner
from backend.app.agents.langgraph.state import AgentState
from backend.app.agents.langgraph.tool_calling import (
    AgentToolCall,
    assistant_tool_calls_payload,
    get_planner_strategy,
)
from backend.app.config.settings import settings
from backend.app.tools import ToolCall, ToolExecutor, ToolResult


class PlannerNode:
    def __init__(self, planner=None) -> None:
        self.legacy_planner = planner

    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        _ensure_state_defaults(state)
        state["step_count"] = int(state.get("step_count", 0)) + 1
        budget_error = check_budget(state, before="llm")
        if budget_error:
            mark_budget_exceeded(state, budget_error)
            _append_trace(
                state,
                event="budget_exceeded",
                node="planner",
                input_data={"query": state["query"]},
                output_data={"termination_reason": budget_error},
                started_at=started_at,
                status="failed",
                error=budget_error,
            )
            return state

        state["llm_call_count"] = int(state.get("llm_call_count", 0)) + 1
        if self.legacy_planner is not None:
            decision = await self._legacy_decision(state)
            state.setdefault("metadata", {})["legacy_planner"] = True
        else:
            decision = await get_planner_strategy().adecide(state)
        state["current_action"] = decision.action
        state["pending_tool_calls"] = [
            tool_call.model_dump() for tool_call in decision.tool_calls
        ]
        if decision.content and decision.action == "final":
            state["final_answer"] = decision.content
        if decision.tool_calls:
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": decision.content or "",
                    "tool_calls": assistant_tool_calls_payload(decision.tool_calls),
                }
            )

        _update_planner_metadata(state, decision.metadata)
        _append_trace(
            state,
            event="planner" if state["metadata"].get("legacy_planner") else "planner_completed",
            node="planner",
            input_data={"query": state["query"]},
            output_data=decision.model_dump(),
            started_at=started_at,
        )
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return asyncio.run(self.acall(state))

    async def _legacy_decision(self, state: AgentState):
        from backend.app.agents.langgraph.tool_calling import AgentDecision, AgentToolCall

        planner = self.legacy_planner
        aplan = getattr(planner, "aplan", None)
        if callable(aplan):
            plan_result = aplan(
                query=state["query"],
                knowledge_base_id=state.get("knowledge_base_id"),
                conversation_id=state.get("conversation_id"),
                memory_context=state.get("memory_context"),
            )
            plan = cast(
                Any,
                await plan_result if isawaitable(plan_result) else plan_result,
            )
        else:
            if planner is None:
                raise RuntimeError("legacy planner is not configured")
            plan = cast(
                Any,
                await asyncio.to_thread(
                    planner.plan,
                    query=state["query"],
                    knowledge_base_id=state.get("knowledge_base_id"),
                    conversation_id=state.get("conversation_id"),
                    memory_context=state.get("memory_context"),
                ),
            )
        state["plan"] = plan.model_dump()
        calls = [
            AgentToolCall(
                id=f"legacy_{index}",
                tool_name=step.tool,
                arguments=step.args,
                index=index,
            )
            for index, step in enumerate(plan.steps)
        ]
        return AgentDecision(
            action="tool_calls" if calls else "final",
            content=None,
            tool_calls=calls,
            metadata={**plan.metadata, "actual_strategy": "json_plan"},
        )


class ToolNode:
    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self.tool_executor = tool_executor or ToolExecutor()

    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        _ensure_state_defaults(state)
        state["step_count"] = int(state.get("step_count", 0)) + 1
        if not state.get("pending_tool_calls"):
            state["pending_tool_calls"] = _pending_from_legacy_plan(state)
        pending = [
            AgentToolCall.model_validate(item)
            for item in state.get("pending_tool_calls", [])
        ]
        if not pending:
            return state

        budget_error = check_budget(state, before="tool")
        if budget_error:
            mark_budget_exceeded(state, budget_error)
            _append_trace(
                state,
                event="budget_exceeded",
                node="tool",
                input_data={"pending_tool_calls": state.get("pending_tool_calls", [])},
                output_data={"termination_reason": budget_error},
                started_at=started_at,
                status="failed",
                error=budget_error,
            )
            return state

        executable_calls: list[AgentToolCall] = []
        for tool_call in pending:
            repeat_error = _update_repeat_guard(state, tool_call)
            if repeat_error:
                state.setdefault("tool_results", []).append(
                    {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.tool_name,
                        "arguments_hash": _arguments_hash(tool_call.arguments),
                        "success": False,
                        "result": None,
                        "error": repeat_error,
                        "metadata": {"same_tool_repeat_limit": True},
                    }
                )
                state["current_action"] = "reflect"
                continue
            executable_calls.append(tool_call)

        remaining = max(settings.AGENT_MAX_TOOL_CALLS - state["tool_call_count"], 0)
        executable_calls = executable_calls[:remaining]
        semaphore = asyncio.Semaphore(settings.AGENT_TOOL_MAX_CONCURRENCY)
        results = await asyncio.gather(
            *[self._execute_tool(state, tool_call, semaphore) for tool_call in executable_calls]
        )
        for tool_call, result, call_started_at in results:
            self._record_result(state, tool_call, result, call_started_at)
        state["pending_tool_calls"] = []
        if not state.get("metadata", {}).get("legacy_planner"):
            _append_trace(
                state,
                event="tool_completed",
                node="tool",
                input_data={"tool_call_count": len(pending)},
                output_data={"executed": len(results)},
                started_at=started_at,
            )
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return asyncio.run(self.acall(state))

    async def _execute_tool(
        self,
        state: AgentState,
        tool_call: AgentToolCall,
        semaphore: asyncio.Semaphore,
    ) -> tuple[AgentToolCall, ToolResult, float]:
        call_started_at = perf_counter()
        async with semaphore:
            executor_call = ToolCall(name=tool_call.tool_name, arguments=tool_call.arguments)
            aexecute = getattr(self.tool_executor, "aexecute", None)
            if callable(aexecute):
                result_value = aexecute(executor_call)
                result = cast(
                    ToolResult,
                    await result_value
                    if isawaitable(result_value)
                    else result_value,
                )
            else:
                result = await asyncio.to_thread(self.tool_executor.execute, executor_call)
        return tool_call, result, call_started_at

    def _record_result(
        self,
        state: AgentState,
        tool_call: AgentToolCall,
        result: ToolResult,
        started_at: float,
    ) -> None:
        state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
        state.setdefault("tool_calls", []).append(
            {
                "id": tool_call.id,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.tool_name,
                "name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "index": tool_call.index,
                "status": "success" if result.success else "failed",
                "duration_ms": result.metadata.get("duration_ms"),
            }
        )
        result_dump = {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.tool_name,
            "arguments_hash": _arguments_hash(tool_call.arguments),
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "metadata": result.metadata,
        }
        state.setdefault("tool_results", []).append(result_dump)
        if tool_call.tool_name == "knowledge_search" and isinstance(result.result, dict):
            state["knowledge"] = result.result
        event = (
            "tool_call"
            if state.get("metadata", {}).get("legacy_planner")
            else ("tool_completed" if result.success else "tool_failed")
        )
        _append_trace(
            state,
            event=event,
            node="tool",
            input_data={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
            output_data=result_dump,
            started_at=started_at,
            status="success" if result.success else "failed",
            error=result.error,
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
        )


class ObservationNode:
    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        _ensure_state_defaults(state)
        state["step_count"] = int(state.get("step_count", 0)) + 1
        observations = []
        for result in state.get("tool_results", []):
            observation = _tool_result_to_observation(result)
            observations.append(observation)
            state["messages"].append(
                {
                    "role": "tool",
                    "name": observation["tool_name"],
                    "tool_call_id": observation["tool_call_id"],
                    "content": observation["content"],
                }
            )
        state["observations"].extend(observations)
        should_reflect, reason = ReflectionGate().should_reflect(state)
        state["current_action"] = "reflect" if should_reflect else "planner"
        if should_reflect:
            state["metadata"].setdefault("reflection", {})["triggered"] = True
            state["metadata"].setdefault("reflection", {})["last_reason"] = reason
        if not state.get("metadata", {}).get("legacy_planner"):
            _append_trace(
                state,
                event="observation_created",
                node="observation",
                input_data={"tool_results": len(state.get("tool_results", []))},
                output_data={"observations": observations, "reflect": should_reflect},
                started_at=started_at,
                status="success",
            )
        state["tool_results"] = []
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return asyncio.run(self.acall(state))


class ReflectionNode:
    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        _ensure_state_defaults(state)
        state["step_count"] = int(state.get("step_count", 0)) + 1
        reason = state["metadata"].get("reflection", {}).get("last_reason") or "unknown"
        budget_error = check_budget(state, before="reflection")
        if budget_error:
            mark_budget_exceeded(state, budget_error)
            return state
        state["reflection_count"] = int(state.get("reflection_count", 0)) + 1
        state["llm_call_count"] = int(state.get("llm_call_count", 0)) + 1
        decision, metadata = await ReflectionNodeRunner().reflect(state, reason)
        state["metadata"].setdefault("reflection", {})["count"] = state["reflection_count"]
        state["metadata"].setdefault("reflection", {})["last_decision"] = decision.model_dump()
        if decision.status in {"final", "fail"}:
            state["current_action"] = "final"
            state["termination_reason"] = (
                "reflection_final" if decision.status == "final" else "reflection_fail"
            )
            state["final_answer"] = decision.final_answer or decision.reason or "Agent 反思后终止。"
        else:
            state["current_action"] = "planner"
            state["messages"].append(
                {
                    "role": "system",
                    "content": (
                        f"Reflection hint: {decision.reason}. "
                        f"Suggested action: {decision.suggested_action}"
                    ),
                }
            )
        _append_trace(
            state,
            event="reflection_completed",
            node="reflection",
            input_data={"reason": reason},
            output_data={"decision": decision.model_dump(), "metadata": metadata},
            started_at=started_at,
            status="success",
        )
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return asyncio.run(self.acall(state))


class FinalNode:
    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        _ensure_state_defaults(state)
        state["step_count"] = int(state.get("step_count", 0)) + 1
        if self._streaming_enabled(state) and self._can_stream_final_answer(state):
            state["final_answer"] = await self._stream_answer(state)
        elif not state.get("final_answer"):
            state["final_answer"] = self._build_answer(state)
        if not state.get("termination_reason"):
            state["termination_reason"] = "final_answer"
        state["loop_status"] = "completed"
        _update_loop_metadata(state, started_at)
        _append_trace(
            state,
            event="final_answer",
            node="final",
            input_data={"query": state["query"]},
            output_data={
                "answer": state.get("final_answer"),
                "termination_reason": state.get("termination_reason"),
            },
            started_at=started_at,
        )
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return asyncio.run(self.acall(state))

    def _streaming_enabled(self, state: AgentState) -> bool:
        return bool(state.get("metadata", {}).get("_agent_stream_answer_enabled"))

    def _can_stream_final_answer(self, state: AgentState) -> bool:
        if state.get("metadata", {}).get("_agent_stream_answer_done"):
            return False
        if state.get("termination_reason") and "exceeded" in str(state.get("termination_reason")):
            return False
        if state.get("metadata", {}).get("reflection", {}).get("last_decision", {}).get(
            "status"
        ) in {"final", "fail"}:
            return False
        return bool(
            state.get("observations")
            or state.get("knowledge")
            or not state.get("final_answer")
        )

    async def _stream_answer(self, state: AgentState) -> str:
        queue = state.get("metadata", {}).get("_agent_stream_event_queue")
        delta_count = 0
        request = build_final_answer_request(
            query=state["query"],
            observations=state.get("observations", []),
            knowledge=state.get("knowledge") if isinstance(state.get("knowledge"), dict) else None,
            fallback_answer=self._build_answer(state),
        )

        async def on_delta(delta: str) -> None:
            nonlocal delta_count
            delta_count += 1
            if queue is not None:
                await queue.put({"event": "answer_delta", "data": {"delta": delta}})

        answer = await collect_streaming_answer(request, on_delta=on_delta)
        state["metadata"]["_agent_stream_answer_done"] = True
        state["metadata"]["answer_stream_delta_count"] = (
            state["metadata"].get("answer_stream_delta_count", 0)
            + delta_count
        )
        return answer or self._build_answer(state)

    def _build_answer(self, state: AgentState) -> str:
        knowledge = state.get("knowledge")
        if isinstance(knowledge, dict) and knowledge.get("answer"):
            return str(knowledge["answer"])
        observations = state.get("observations", [])
        failures = [item for item in observations if not item.get("success", True)]
        if observations and len(failures) == len(observations):
            return "工具调用失败，Agent 已停止。"
        if state.get("termination_reason") and "exceeded" in str(state.get("termination_reason")):
            return f"Agent 执行预算已用尽：{state.get('termination_reason')}。"
        if observations:
            if (
                state.get("metadata", {})
                .get("agent_loop", {})
                .get("planner_strategy")
                == "json_plan"
                and len(observations) == 1
            ):
                return str(observations[0].get("raw_result") or observations[0].get("content", ""))
            return "\n".join(
                str(item.get("content", "")) for item in observations if item.get("content")
            )
        return "当前 Agent 未生成最终回答。"


def route_after_planner(state: AgentState) -> str:
    if state.get("current_action") == "tool_calls":
        return "tool"
    if state.get("current_action") == "reflect":
        return "reflection"
    return "final"


def route_after_observation(state: AgentState) -> str:
    if state.get("metadata", {}).get("legacy_planner"):
        return "final"
    if (
        state.get("metadata", {})
        .get("agent_loop", {})
        .get("planner_strategy")
        == "json_plan"
    ):
        return "final"
    if state.get("current_action") == "reflect":
        return "reflection"
    if state.get("current_action") == "final":
        return "final"
    return "planner"


def _ensure_state_defaults(state: AgentState) -> None:
    state.setdefault("messages", [])
    state.setdefault("tool_calls", [])
    state.setdefault("tool_results", [])
    state.setdefault("observations", [])
    state.setdefault("pending_tool_calls", [])
    state.setdefault("metadata", {})
    state["metadata"].setdefault("trace", [])
    state["metadata"].setdefault("agent_loop", {})
    state["metadata"].setdefault("tool_calling", {})
    state["metadata"].setdefault("reflection", {})
    state.setdefault("step_count", 0)
    state.setdefault("llm_call_count", 0)
    state.setdefault("tool_call_count", 0)
    state.setdefault("reflection_count", 0)
    state.setdefault("same_tool_repeat_count", 0)
    state.setdefault("last_tool_name", None)
    state.setdefault("last_tool_arguments_hash", None)
    state.setdefault("current_action", "planner")
    state.setdefault("loop_status", "running")
    state.setdefault("termination_reason", None)
    state.setdefault("budget", AgentExecutionBudget.from_settings().model_dump())


def route_after_reflection(state: AgentState) -> str:
    return "final" if state.get("current_action") == "final" else "planner"


def _tool_result_to_observation(result: dict) -> dict:
    content = result.get("result")
    if isinstance(content, dict):
        content_text = json.dumps(content, ensure_ascii=False)
    else:
        content_text = str(content or result.get("error") or "")
    if len(content_text) > settings.AGENT_OBSERVATION_MAX_CHARS:
        content_text = content_text[: settings.AGENT_OBSERVATION_MAX_CHARS] + "...[truncated]"
    return {
        "tool_call_id": result.get("tool_call_id"),
        "tool_name": result.get("tool_name"),
        "success": result.get("success", False),
        "content": content_text,
        "raw_result": content,
        "metadata": _sanitize_metadata(result.get("metadata", {})),
        "error": result.get("error"),
    }


def _pending_from_legacy_plan(state: AgentState) -> list[dict]:
    plan = state.get("plan", {})
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    pending = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        args = step.get("args")
        if isinstance(tool_name, str) and isinstance(args, dict):
            pending.append(
                AgentToolCall(
                    id=f"legacy_plan_{index}",
                    tool_name=tool_name,
                    arguments=args,
                    index=index,
                ).model_dump()
            )
    return pending


def _update_repeat_guard(state: AgentState, tool_call: AgentToolCall) -> str | None:
    arguments_hash = _arguments_hash(tool_call.arguments)
    if state.get("last_tool_name") == tool_call.tool_name and state.get(
        "last_tool_arguments_hash"
    ) == arguments_hash:
        state["same_tool_repeat_count"] += 1
    else:
        state["same_tool_repeat_count"] = 1
    state["last_tool_name"] = tool_call.tool_name
    state["last_tool_arguments_hash"] = arguments_hash
    if state["same_tool_repeat_count"] > settings.AGENT_MAX_SAME_TOOL_REPEATS:
        state["termination_reason"] = "same_tool_repeat_limit"
        state["metadata"].setdefault("agent_loop", {})[
            "same_tool_repeat_limit_triggered"
        ] = True
        return "same tool repeat limit reached"
    return None


def _update_planner_metadata(state: AgentState, metadata: dict) -> None:
    tool_registry = metadata.get("tool_registry", {})
    state["metadata"].setdefault("agent_loop", {})["planner_strategy"] = metadata.get(
        "actual_strategy",
        settings.AGENT_PLANNER_STRATEGY,
    )
    state["metadata"].setdefault("agent_loop", {})["planner_fallback_used"] = metadata.get(
        "fallback_used",
        False,
    )
    state["metadata"].setdefault("tool_calling", {}).update(
        {
            "native_supported": metadata.get("actual_strategy") == "native_tool_calling",
            "native_used": metadata.get("actual_strategy") == "native_tool_calling",
            "available_tool_count": tool_registry.get("available_tool_count", 0),
            "registry_version": tool_registry.get("registry_version", 0),
        }
    )
    state["metadata"]["tool_registry"] = tool_registry


def _update_loop_metadata(state: AgentState, started_at: float) -> None:
    metadata = state["metadata"].setdefault("agent_loop", {})
    metadata.update(
        {
            "enabled": settings.AGENT_LOOP_ENABLED,
            "steps": state.get("step_count", 0),
            "llm_calls": state.get("llm_call_count", 0),
            "tool_calls": state.get("tool_call_count", 0),
            "reflections": state.get("reflection_count", 0),
            "loop_iterations": state.get("llm_call_count", 0),
            "termination_reason": state.get("termination_reason"),
            "duration_ms": round((perf_counter() - float(state["budget"]["started_at"])) * 1000, 2),
        }
    )


def _append_trace(
    state: AgentState,
    *,
    event: str,
    node: str,
    input_data: dict,
    output_data: dict,
    started_at: float,
    status: str = "success",
    error: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> None:
    state["metadata"].setdefault("trace", []).append(
        {
            "step": event,
            "name": node,
            "node": node,
            "event": event,
            "input": input_data,
            "output": output_data,
            "input_summary": _summary(input_data),
            "output_summary": _summary(output_data),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "status": status,
            "error": error,
            "llm_call_count": state.get("llm_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "reflection_count": state.get("reflection_count", 0),
            "budget_remaining": budget_remaining(state),
            "async_execution": True,
        }
    )


def _arguments_hash(arguments: dict) -> str:
    raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _summary(value: dict) -> dict:
    raw = json.dumps(value, ensure_ascii=False, default=str)
    if len(raw) > 500:
        raw = raw[:500] + "...[truncated]"
    return {"preview": raw}


def _sanitize_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if not any(token in key.lower() for token in ["key", "token", "secret", "password"])
    }
