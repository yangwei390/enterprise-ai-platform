import asyncio
import hashlib
from time import perf_counter

from backend.app.agents.langgraph.graph import LangGraphUnavailable, build_agent_graph
from backend.app.agents.langgraph.state import AgentState, create_initial_state
from backend.app.agents.runtime import AgentRuntime
from backend.app.agents.state import AgentRuntimeRequest, AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.memory.factory import MemoryFactory
from backend.app.memory.state import MemoryState


class LangGraphAgentRuntime:
    def __init__(
        self,
        fallback_runtime: AgentRuntime | None = None,
        graph_app=None,
    ) -> None:
        self.fallback_runtime = fallback_runtime or AgentRuntime()
        self.graph_app = graph_app

    def run(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        try:
            graph_app = self.graph_app or build_agent_graph()
            session_id = self._session_id(request)
            session_state = self._load_session(session_id)
            state = create_initial_state(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                memory_context=request.memory_context,
                metadata=self._metadata_with_session(
                    request.metadata,
                    session_id=session_id,
                    session_state=session_state,
                ),
            )
            self._inject_session_state(state, session_state)
            result_state = self._invoke_graph(graph_app, state)
            self._save_session(session_id, result_state)
            return self._to_result(result_state)
        except LangGraphUnavailable as exc:
            logger.warning(f"LangGraph unavailable, fallback to AgentRuntime(V1): {exc}")
            result = self.fallback_runtime.run(request)
            result.metadata["runtime_fallback"] = "v1"
            result.metadata["runtime_fallback_reason"] = str(exc)
            return result
        except Exception as exc:
            logger.exception("LangGraph runtime failed, fallback to AgentRuntime(V1)")
            result = self.fallback_runtime.run(request)
            result.metadata["runtime_fallback"] = "v1"
            result.metadata["runtime_fallback_reason"] = str(exc)
            return result

    async def arun(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        started_at = perf_counter()
        async_metadata = {
            "enabled": settings.AGENT_ASYNC_ENABLED,
            "runtime": "langgraph",
            "duration_ms": 0,
            "timeout_seconds": settings.AGENT_ASYNC_TIMEOUT_SECONDS,
            "timed_out": False,
            "cancelled": False,
            "sync_fallback_used": False,
            "failed": False,
            "error": None,
            "tool_concurrency": settings.AGENT_TOOL_MAX_CONCURRENCY,
            "total_tool_calls": 0,
            "async_tool_calls": 0,
            "sync_fallback_calls": 0,
            "timed_out_calls": 0,
            "failed_calls": 0,
            "retry_calls": 0,
        }

        if not settings.AGENT_ASYNC_ENABLED:
            result = await asyncio.to_thread(self.run, request)
            result.metadata["async_runtime"] = {
                **async_metadata,
                "enabled": False,
                "sync_fallback_used": True,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
            return result

        try:
            graph_app = self.graph_app or build_agent_graph(async_mode=True)
            session_id = self._session_id(request)
            session_state = self._load_session(session_id)
            state = create_initial_state(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                memory_context=request.memory_context,
                metadata=self._metadata_with_session(
                    {
                        **request.metadata,
                        "async_runtime": async_metadata,
                    },
                    session_id=session_id,
                    session_state=session_state,
                ),
            )
            self._inject_session_state(state, session_state)
            async with asyncio.timeout(settings.AGENT_ASYNC_TIMEOUT_SECONDS):
                result_state = await self._ainvoke_graph(graph_app, state)
            self._save_session(session_id, result_state)
            result = self._to_result(result_state)
            runtime_metadata = result.metadata.get("async_runtime", {})
            runtime_metadata.update(
                {
                    "enabled": True,
                    "runtime": "langgraph",
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "timeout_seconds": settings.AGENT_ASYNC_TIMEOUT_SECONDS,
                    "timed_out": False,
                    "cancelled": False,
                    "failed": False,
                    "error": None,
                    "sync_fallback_used": runtime_metadata.get(
                        "sync_fallback_calls", 0
                    )
                    > 0,
                }
            )
            result.metadata["async_runtime"] = runtime_metadata
            return result
        except asyncio.CancelledError:
            logger.info("LangGraph async runtime cancelled")
            raise
        except TimeoutError as exc:
            return await self._fallback_from_async_error(
                request=request,
                started_at=started_at,
                async_metadata={
                    **async_metadata,
                    "timed_out": True,
                    "failed": True,
                    "error": str(exc) or "agent async runtime timed out",
                },
                reason="agent async runtime timed out",
            )
        except LangGraphUnavailable as exc:
            logger.warning(f"LangGraph unavailable, fallback to AgentRuntime(V1): {exc}")
            return await self._fallback_from_async_error(
                request=request,
                started_at=started_at,
                async_metadata={
                    **async_metadata,
                    "failed": True,
                    "error": str(exc),
                },
                reason=str(exc),
            )
        except Exception as exc:
            logger.exception("LangGraph async runtime failed")
            return await self._fallback_from_async_error(
                request=request,
                started_at=started_at,
                async_metadata={
                    **async_metadata,
                    "failed": True,
                    "error": str(exc),
                },
                reason=str(exc),
            )

    async def _fallback_from_async_error(
        self,
        *,
        request: AgentRuntimeRequest,
        started_at: float,
        async_metadata: dict,
        reason: str,
    ) -> AgentRuntimeResult:
        async_metadata["duration_ms"] = round((perf_counter() - started_at) * 1000, 2)
        if not settings.AGENT_ASYNC_FAIL_OPEN:
            return AgentRuntimeResult(
                answer=f"Agent 异步执行失败：{reason}",
                action="failed",
                metadata={"async_runtime": async_metadata},
                trace=[],
            )

        result = await asyncio.to_thread(self.fallback_runtime.run, request)
        result.metadata["runtime_fallback"] = "v1"
        result.metadata["runtime_fallback_reason"] = reason
        result.metadata["async_runtime"] = {
            **async_metadata,
            "sync_fallback_used": True,
        }
        return result

    def _invoke_graph(self, graph_app, state: AgentState) -> dict:
        try:
            return graph_app.invoke(
                state,
                config={"recursion_limit": settings.AGENT_MAX_STEPS + 8},
            )
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            return graph_app.invoke(state)

    async def _ainvoke_graph(self, graph_app, state: AgentState) -> dict:
        try:
            return await graph_app.ainvoke(
                state,
                config={"recursion_limit": settings.AGENT_MAX_STEPS + 8},
            )
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            return await graph_app.ainvoke(state)

    def _session_id(self, request: AgentRuntimeRequest) -> str:
        metadata_session_id = request.metadata.get("session_id")
        if metadata_session_id:
            return str(metadata_session_id)
        if request.conversation_id is not None:
            return f"conversation:{request.conversation_id}"
        digest = hashlib.sha256(request.query.encode("utf-8")).hexdigest()[:16]
        return f"agent:{digest}"

    def _load_session(self, session_id: str) -> MemoryState | None:
        try:
            return MemoryFactory.get_manager().load_session(session_id)
        except Exception as exc:
            logger.warning(f"Agent session memory load failed | session={session_id}: {exc}")
            return None

    def _metadata_with_session(
        self,
        metadata: dict,
        *,
        session_id: str,
        session_state: MemoryState | None,
    ) -> dict:
        return {
            **metadata,
            "memory": {
                **metadata.get("memory", {}),
                "provider": MemoryFactory.get_manager().provider.name,
                "session_loaded": session_state is not None,
            },
            "session": {
                "session_id": session_id,
                "loaded": session_state is not None,
            },
        }

    def _inject_session_state(
        self,
        state: AgentState,
        session_state: MemoryState | None,
    ) -> None:
        if session_state is None:
            return
        restored_messages = session_state.messages[-settings.AGENT_MEMORY_MAX_LOOP_MESSAGES :]
        state["messages"] = [*restored_messages, *state.get("messages", [])]
        state["metadata"]["session"]["restored_tool_result_count"] = len(
            session_state.tool_results
        )
        state["metadata"]["session"]["trace_id"] = session_state.trace_id

    def _save_session(self, session_id: str, state: dict) -> None:
        try:
            session_state = MemoryState(
                session_id=session_id,
                messages=state.get("messages", [])[-settings.AGENT_MEMORY_MAX_LOOP_MESSAGES :],
                tool_results=state.get("observations", [])[
                    -settings.AGENT_MEMORY_MAX_LOOP_MESSAGES :
                ],
                current_plan=state.get("plan"),
                current_step=str(state.get("current_action") or "final"),
                planner_output=state.get("plan"),
                workflow_state={},
                trace_id=state.get("metadata", {}).get("trace_id"),
                session_metadata={
                    "runtime": "langgraph",
                    "final_answer": state.get("final_answer"),
                    "termination_reason": state.get("termination_reason"),
                    "step_count": state.get("step_count"),
                },
            )
            MemoryFactory.get_manager().save_session(session_state)
            state.setdefault("metadata", {}).setdefault("memory", {})[
                "session_saved"
            ] = True
            state.setdefault("metadata", {}).setdefault("checkpoint", {})[
                "enabled"
            ] = True
        except Exception as exc:
            logger.warning(f"Agent session memory save failed | session={session_id}: {exc}")
            state.setdefault("metadata", {}).setdefault("memory", {})[
                "session_saved"
            ] = False
            state.setdefault("metadata", {}).setdefault("memory", {})[
                "error"
            ] = str(exc)

    def _to_result(self, state: dict) -> AgentRuntimeResult:
        metadata = dict(state.get("metadata", {}))
        trace = [
            AgentTraceStep.model_validate(item)
            for item in metadata.pop("trace", [])
        ]
        knowledge = state.get("knowledge")
        sources = []
        citations = []
        if isinstance(knowledge, dict):
            sources = knowledge.get("sources", [])
            citations = knowledge.get("citations", [])
            metadata["knowledge_metadata"] = knowledge.get("metadata", {})

        return AgentRuntimeResult(
            answer=str(state.get("final_answer") or ""),
            action="tool" if state.get("tool_calls") else "direct_answer",
            tool_calls=state.get("tool_calls", []),
            observations=state.get("observations", []),
            sources=sources,
            citations=citations,
            metadata=metadata,
            trace=trace,
        )
