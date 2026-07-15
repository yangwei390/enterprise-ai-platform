import re
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Protocol

from backend.app.agents.final_answer import (
    build_final_answer_request,
    stream_final_answer,
)
from backend.app.agents.state import (
    AgentRuntimeRequest,
    AgentRuntimeResult,
    AgentState,
    PlannerDecision,
)
from backend.app.agents.trace import AgentTraceStep
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.tools import ToolCall, ToolExecutor, ToolResult


class ToolExecutorProtocol(Protocol):
    def execute(self, tool_call: ToolCall) -> ToolResult:
        ...


class SimplePlanner:
    def plan(self, request: AgentRuntimeRequest) -> PlannerDecision:
        query = request.query.strip()
        if self._looks_like_knowledge_query(query):
            return PlannerDecision(
                action="tool",
                tool_name="knowledge_search",
                reason="query looks like knowledge-base question",
                confidence=0.9,
            )
        if self._looks_like_time_query(query):
            return PlannerDecision(
                action="tool",
                tool_name="get_current_time",
                reason="query asks for current time",
                confidence=0.85,
            )
        if self._looks_like_calculation(query):
            return PlannerDecision(
                action="tool",
                tool_name="calculator",
                reason="query contains arithmetic expression",
                confidence=0.85,
            )
        return PlannerDecision(
            action="direct_answer",
            tool_name=None,
            reason="no tool rule matched",
            confidence=0.5,
        )

    def build_tool_arguments(
        self,
        request: AgentRuntimeRequest,
        decision: PlannerDecision,
    ) -> dict:
        if decision.tool_name == "knowledge_search":
            return {
                "query": request.query,
                "knowledge_base_id": request.knowledge_base_id,
                "conversation_id": request.conversation_id,
                "memory_context": request.memory_context,
            }
        if decision.tool_name == "calculator":
            return {"expression": self._extract_expression(request.query)}
        if decision.tool_name == "get_current_time":
            return {"timezone": None}
        if decision.tool_name == "echo":
            return {"text": request.query}
        return {}

    def _looks_like_knowledge_query(self, query: str) -> bool:
        if re.search(r"第[0-9一二三四五六七八九十百千万几]+[章节条]", query):
            return True
        keywords = [
            "知识库",
            "文档",
            "根据资料",
            "rag",
            "什么是",
            "是什么",
            "解释",
            "总结",
            "说的什么",
        ]
        lowered_query = query.lower()
        return any(keyword in lowered_query for keyword in keywords)

    def _looks_like_time_query(self, query: str) -> bool:
        lowered_query = query.lower()
        return any(keyword in lowered_query for keyword in ["现在几点", "当前时间", "time"])

    def _looks_like_calculation(self, query: str) -> bool:
        if any(keyword in query for keyword in ["加", "减", "乘", "除"]):
            return True
        return bool(re.search(r"\d+\s*[+\-*/%]\s*\d+", query))

    def _extract_expression(self, query: str) -> str:
        allowed_parts = re.findall(r"[0-9+\-*/%().\s]+", query)
        expression = "".join(allowed_parts).strip()
        return expression or query


class AgentRuntime:
    def __init__(
        self,
        planner: SimplePlanner | None = None,
        tool_executor: ToolExecutorProtocol | None = None,
    ) -> None:
        self.planner = planner or SimplePlanner()
        self.tool_executor = tool_executor or ToolExecutor()

    async def astream_events(
        self,
        request: AgentRuntimeRequest,
    ) -> AsyncIterator[dict]:
        state = AgentState(
            query=request.query,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
            memory_context=request.memory_context,
            metadata={**request.metadata, "answer_streamed": True},
        )
        trace: list[AgentTraceStep] = []

        yield {
            "event": "status",
            "data": {"status": "analyzing", "message": "正在分析问题"},
        }
        decision = self._run_planner(request, state, trace)
        if decision.action == "tool" and decision.tool_name:
            yield {"event": "status", "data": self._status_from_tool(decision.tool_name)}
            self._run_tool(request, state, decision, trace)
        else:
            yield {
                "event": "status",
                "data": {"status": "answering", "message": "正在整理答案"},
            }

        answer_parts: list[str] = []
        if state.errors:
            if state.final_answer:
                answer_parts.append(state.final_answer)
                yield {"event": "answer_delta", "data": {"delta": state.final_answer}}
        else:
            yield {
                "event": "status",
                "data": {"status": "answering", "message": "正在整理答案"},
            }
            final_request = build_final_answer_request(
                query=request.query,
                observations=state.observations,
                fallback_answer=state.final_answer,
            )
            async for delta in stream_final_answer(final_request):
                answer_parts.append(delta)
                yield {"event": "answer_delta", "data": {"delta": delta}}

        state.final_answer = "".join(answer_parts) or state.final_answer or ""
        state.metadata["answer_stream_delta_count"] = len(answer_parts)
        self._record_final_answer(state, trace)
        yield {
            "event": "result",
            "data": {
                "result": AgentRuntimeResult(
                    answer=state.final_answer,
                    action=decision.action,
                    tool_calls=state.tool_calls,
                    observations=state.observations,
                    sources=state.metadata.get("sources", []),
                    citations=state.metadata.get("citations", []),
                    metadata=state.metadata,
                    trace=trace,
                ).model_dump()
            },
        }

    def run(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        state = AgentState(
            query=request.query,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
            memory_context=request.memory_context,
            metadata=dict(request.metadata),
        )
        trace: list[AgentTraceStep] = []

        decision = self._run_planner(request, state, trace)
        if decision.action == "tool" and decision.tool_name:
            self._run_tool(request, state, decision, trace)
        else:
            self._run_direct_answer(request, state, trace)

        self._record_final_answer(state, trace)
        return AgentRuntimeResult(
            answer=state.final_answer or "",
            action=decision.action,
            tool_calls=state.tool_calls,
            observations=state.observations,
            sources=state.metadata.get("sources", []),
            citations=state.metadata.get("citations", []),
            metadata=state.metadata,
            trace=trace,
        )

    def _status_from_tool(self, tool_name: str) -> dict:
        if tool_name == "knowledge_search":
            return {"status": "retrieving", "message": "正在查询知识库"}
        return {"status": "processing", "message": "正在处理任务"}

    def _run_planner(
        self,
        request: AgentRuntimeRequest,
        state: AgentState,
        trace: list[AgentTraceStep],
    ) -> PlannerDecision:
        started_at = perf_counter()
        try:
            decision = self.planner.plan(request)
            state.planner_decision = decision
            self._append_trace(
                trace,
                step="planner",
                name="simple_planner",
                input_data={"query": request.query},
                output_data=decision.model_dump(),
                started_at=started_at,
            )
            return decision
        except Exception as exc:
            state.errors.append(str(exc))
            decision = PlannerDecision(
                action="direct_answer",
                tool_name=None,
                reason="planner failed",
                confidence=0,
            )
            state.planner_decision = decision
            self._append_trace(
                trace,
                step="planner",
                name="simple_planner",
                input_data={"query": request.query},
                output_data=decision.model_dump(),
                started_at=started_at,
                status="failed",
                error=str(exc),
            )
            return decision

    def _run_tool(
        self,
        request: AgentRuntimeRequest,
        state: AgentState,
        decision: PlannerDecision,
        trace: list[AgentTraceStep],
    ) -> None:
        tool_arguments = self.planner.build_tool_arguments(request, decision)
        tool_call = ToolCall(name=str(decision.tool_name), arguments=tool_arguments)
        state.tool_calls.append(tool_call.model_dump())

        started_at = perf_counter()
        try:
            tool_result = self.tool_executor.execute(tool_call)
        except Exception as exc:
            state.errors.append(str(exc))
            state.final_answer = f"工具调用失败：{exc}"
            self._append_trace(
                trace,
                step="tool_call",
                name=tool_call.name,
                input_data=tool_call.model_dump(),
                output_data={},
                started_at=started_at,
                status="failed",
                error=str(exc),
            )
            return

        state.observations.append(tool_result.model_dump())
        self._append_trace(
            trace,
            step="tool_call",
            name=tool_call.name,
            input_data=tool_call.model_dump(),
            output_data=tool_result.model_dump(),
            started_at=started_at,
            status="success" if tool_result.success else "failed",
            error=tool_result.error,
        )
        self._append_trace(
            trace,
            step="observation",
            name=tool_call.name,
            input_data={"tool": tool_call.name},
            output_data=tool_result.model_dump(),
            started_at=started_at,
            status="success" if tool_result.success else "failed",
            error=tool_result.error,
        )
        self._build_answer_from_tool_result(state, tool_result)

    def _run_direct_answer(
        self,
        request: AgentRuntimeRequest,
        state: AgentState,
        trace: list[AgentTraceStep],
    ) -> None:
        started_at = perf_counter()
        try:
            llm_response = LLMFactory.get_llm().chat(
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role="user",
                            content=request.query,
                        )
                    ],
                    metadata={"agent_runtime": True},
                )
            )
            state.final_answer = llm_response.answer
            state.metadata["llm_model"] = llm_response.model
            state.metadata["llm_metadata"] = llm_response.metadata
            self._append_trace(
                trace,
                step="direct_answer",
                name="llm",
                input_data={"query": request.query},
                output_data={"answer": llm_response.answer, "model": llm_response.model},
                started_at=started_at,
            )
        except Exception as exc:
            state.errors.append(str(exc))
            state.final_answer = "当前 Agent V1 未调用工具，直接回答能力待增强。"
            self._append_trace(
                trace,
                step="direct_answer",
                name="llm",
                input_data={"query": request.query},
                output_data={"answer": state.final_answer},
                started_at=started_at,
                status="failed",
                error=str(exc),
            )

    def _build_answer_from_tool_result(
        self,
        state: AgentState,
        tool_result: ToolResult,
    ) -> None:
        if not tool_result.success:
            state.final_answer = f"工具调用失败：{tool_result.error or 'unknown error'}"
            return

        if tool_result.name == "knowledge_search" and isinstance(tool_result.result, dict):
            state.final_answer = str(tool_result.result.get("answer") or "")
            state.metadata["sources"] = tool_result.result.get("sources", [])
            state.metadata["citations"] = tool_result.result.get("citations", [])
            state.metadata["knowledge_metadata"] = tool_result.result.get("metadata", {})
            return

        if tool_result.name == "calculator" and isinstance(tool_result.result, dict):
            state.final_answer = f"计算结果是：{tool_result.result.get('value')}"
            return

        if tool_result.name == "get_current_time" and isinstance(tool_result.result, dict):
            state.final_answer = f"当前时间是：{tool_result.result.get('time')}"
            return

        if tool_result.name == "echo" and isinstance(tool_result.result, dict):
            state.final_answer = str(tool_result.result.get("text") or "")
            return

        state.final_answer = str(tool_result.result or "")

    def _record_final_answer(
        self,
        state: AgentState,
        trace: list[AgentTraceStep],
    ) -> None:
        started_at = perf_counter()
        self._append_trace(
            trace,
            step="final_answer",
            name="final_answer",
            input_data={"query": state.query},
            output_data={"answer": state.final_answer},
            started_at=started_at,
            status="success" if not state.errors else "failed",
            error="; ".join(state.errors) if state.errors else None,
        )
        state.metadata["errors"] = state.errors

    def _append_trace(
        self,
        trace: list[AgentTraceStep],
        step: str,
        name: str,
        input_data: dict,
        output_data: dict,
        started_at: float,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        trace.append(
            AgentTraceStep(
                step=step,
                name=name,
                input=input_data,
                output=output_data,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                status=status,
                error=error,
            )
        )
