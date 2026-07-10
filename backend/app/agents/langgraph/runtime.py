import asyncio
from time import perf_counter

from backend.app.agents.langgraph.graph import LangGraphUnavailable, build_agent_graph
from backend.app.agents.langgraph.state import create_initial_state
from backend.app.agents.runtime import AgentRuntime
from backend.app.agents.state import AgentRuntimeRequest, AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
from backend.app.config.settings import settings
from backend.app.logger import logger


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
            state = create_initial_state(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                memory_context=request.memory_context,
                metadata=request.metadata,
            )
            result_state = graph_app.invoke(state)
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
            state = create_initial_state(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                memory_context=request.memory_context,
                metadata={
                    **request.metadata,
                    "async_runtime": async_metadata,
                },
            )
            async with asyncio.timeout(settings.AGENT_ASYNC_TIMEOUT_SECONDS):
                result_state = await graph_app.ainvoke(state)
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
            observations=state.get("tool_results", []),
            sources=sources,
            citations=citations,
            metadata=metadata,
            trace=trace,
        )
