import asyncio
from time import perf_counter

from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.langgraph.state import AgentState
from backend.app.config.settings import settings
from backend.app.tools import ToolCall, ToolExecutor, ToolResult


class PlannerNode:
    def __init__(self, planner: LLMPlanner | None = None) -> None:
        self.planner = planner or LLMPlanner()

    def __call__(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        plan = self.planner.plan(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id"),
            conversation_id=state.get("conversation_id"),
            memory_context=state.get("memory_context"),
        )
        state["plan"] = plan.model_dump()
        _append_trace(
            state,
            step="planner",
            name="llm_planner",
            input_data={"query": state["query"]},
            output_data=state["plan"],
            started_at=started_at,
        )
        return state

    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        plan = await self.planner.aplan(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id"),
            conversation_id=state.get("conversation_id"),
            memory_context=state.get("memory_context"),
        )
        state["plan"] = plan.model_dump()
        _append_trace(
            state,
            step="planner",
            name="llm_planner",
            input_data={"query": state["query"]},
            output_data=state["plan"],
            started_at=started_at,
            extra={"async_execution": True, "sync_fallback": True},
        )
        return state


class ToolNode:
    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self.tool_executor = tool_executor or ToolExecutor()

    def __call__(self, state: AgentState) -> AgentState:
        plan = state.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        if not steps:
            return state

        for step in steps:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            args = step.get("args")
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                continue

            tool_call = ToolCall(name=tool_name, arguments=args)
            state["tool_calls"].append(tool_call.model_dump())
            started_at = perf_counter()
            result = self.tool_executor.execute(tool_call)
            result_dump = result.model_dump()
            state["tool_results"].append(result_dump)
            if result.name == "knowledge_search" and isinstance(result.result, dict):
                state["knowledge"] = result.result
            _append_trace(
                state,
                step="tool_call",
                name=tool_name,
                input_data=tool_call.model_dump(),
                output_data=result_dump,
                started_at=started_at,
                status="success" if result.success else "failed",
                error=result.error,
            )
        return state

    async def acall(self, state: AgentState) -> AgentState:
        plan = state.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        if not steps:
            return state

        valid_steps = [
            step
            for step in steps
            if isinstance(step, dict)
            and isinstance(step.get("tool"), str)
            and isinstance(step.get("args"), dict)
        ]
        if not valid_steps:
            return state

        if any(step.get("depends_on") for step in valid_steps):
            for index, step in enumerate(valid_steps):
                await self._execute_step(state=state, step=step, index=index)
            return state

        semaphore = asyncio.Semaphore(settings.AGENT_TOOL_MAX_CONCURRENCY)
        results = await asyncio.gather(
            *[
                self._execute_step(
                    state=state,
                    step=step,
                    index=index,
                    semaphore=semaphore,
                )
                for index, step in enumerate(valid_steps)
            ]
        )
        results.sort(key=lambda item: item[0])
        for _, tool_call, result, started_at in results:
            self._record_tool_result(
                state=state,
                tool_call=tool_call,
                result=result,
                started_at=started_at,
            )
        return state

    async def _execute_step(
        self,
        *,
        state: AgentState,
        step: dict,
        index: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[int, ToolCall, ToolResult, float]:
        tool_call = ToolCall(name=str(step["tool"]), arguments=dict(step["args"]))
        started_at = perf_counter()
        if semaphore is None:
            result = await self.tool_executor.aexecute(tool_call)
        else:
            async with semaphore:
                result = await self.tool_executor.aexecute(tool_call)
        if semaphore is None:
            self._record_tool_result(
                state=state,
                tool_call=tool_call,
                result=result,
                started_at=started_at,
            )
        return index, tool_call, result, started_at

    def _record_tool_result(
        self,
        *,
        state: AgentState,
        tool_call: ToolCall,
        result: ToolResult,
        started_at: float,
    ) -> None:
        state["tool_calls"].append(tool_call.model_dump())
        result_dump = result.model_dump()
        state["tool_results"].append(result_dump)
        if result.name == "knowledge_search" and isinstance(result.result, dict):
            state["knowledge"] = result.result

        async_runtime = state["metadata"].setdefault("async_runtime", {})
        async_runtime["total_tool_calls"] = async_runtime.get("total_tool_calls", 0) + 1
        if result.metadata.get("async_execution"):
            async_runtime["async_tool_calls"] = async_runtime.get("async_tool_calls", 0) + 1
        if result.metadata.get("sync_fallback"):
            async_runtime["sync_fallback_calls"] = (
                async_runtime.get("sync_fallback_calls", 0) + 1
            )
        if result.metadata.get("timeout"):
            async_runtime["timed_out_calls"] = async_runtime.get("timed_out_calls", 0) + 1
        if not result.success:
            async_runtime["failed_calls"] = async_runtime.get("failed_calls", 0) + 1
        async_runtime["retry_calls"] = async_runtime.get("retry_calls", 0) + int(
            result.metadata.get("retry_count", 0)
        )

        _append_trace(
            state,
            step="tool_call",
            name=tool_call.name,
            input_data=tool_call.model_dump(),
            output_data=result_dump,
            started_at=started_at,
            status="success" if result.success else "failed",
            error=result.error,
            extra={
                "async_execution": result.metadata.get("async_execution", True),
                "duration_ms": result.metadata.get("duration_ms"),
                "timeout": result.metadata.get("timeout", False),
                "cancelled": result.metadata.get("cancelled", False),
                "retry_count": result.metadata.get("retry_count", 0),
                "sync_fallback": result.metadata.get("sync_fallback", False),
            },
        )


class FinalNode:
    def __call__(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        answer = self._build_answer(state)
        state["final_answer"] = answer
        _append_trace(
            state,
            step="final_answer",
            name="final_answer",
            input_data={"query": state["query"]},
            output_data={"answer": answer},
            started_at=started_at,
        )
        return state

    async def acall(self, state: AgentState) -> AgentState:
        started_at = perf_counter()
        answer = self._build_answer(state)
        state["final_answer"] = answer
        _append_trace(
            state,
            step="final_answer",
            name="final_answer",
            input_data={"query": state["query"]},
            output_data={"answer": answer},
            started_at=started_at,
            extra={"async_execution": True},
        )
        return state

    def _build_answer(self, state: AgentState) -> str:
        knowledge = state.get("knowledge")
        if isinstance(knowledge, dict):
            answer = knowledge.get("answer")
            if answer:
                return str(answer)

        for result in state["tool_results"]:
            tool_result = ToolResult.model_validate(result)
            if not tool_result.success:
                return f"工具调用失败：{tool_result.error or 'unknown error'}"
            if tool_result.result:
                return str(tool_result.result)

        return "当前 LangGraph Runtime 未调用工具，直接回答能力待增强。"


def _append_trace(
    state: AgentState,
    *,
    step: str,
    name: str,
    input_data: dict,
    output_data: dict,
    started_at: float,
    status: str = "success",
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    trace = state["metadata"].setdefault("trace", [])
    item = {
        "step": step,
        "name": name,
        "input": input_data,
        "output": output_data,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "status": status,
        "error": error,
    }
    if extra:
        item.update(extra)
    trace.append(item)
