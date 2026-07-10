from time import perf_counter

from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.langgraph.state import AgentState
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
) -> None:
    trace = state["metadata"].setdefault("trace", [])
    trace.append(
        {
            "step": step,
            "name": name,
            "input": input_data,
            "output": output_data,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "status": status,
            "error": error,
        }
    )
