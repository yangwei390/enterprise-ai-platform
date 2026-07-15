import asyncio
import hashlib
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from backend.app.agents.definition import (
    AgentDefinition,
    AgentDefinitionError,
    get_agent_definition_registry,
)
from backend.app.agents.langgraph.graph import LangGraphUnavailable, build_agent_graph
from backend.app.agents.langgraph.state import AgentState, create_initial_state
from backend.app.agents.state import AgentRuntimeRequest, AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.memory.factory import MemoryFactory
from backend.app.memory.state import MemoryState


class LangGraphAgentRuntime:
    def __init__(
        self,
        graph_app=None,
    ) -> None:
        self.graph_app = graph_app

    def run(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        started_at = perf_counter()
        try:
            definition = self._load_definition(request)
            graph_app = self.graph_app or build_agent_graph()
            session_id = self._session_id(request)
            session_state = self._load_session(session_id)
            state = self._create_state(
                request=request,
                definition=definition,
                session_id=session_id,
                session_state=session_state,
            )
            self._inject_session_state(state, session_state)
            result_state = self._invoke_graph(graph_app, state)
            self._save_session(session_id, result_state)
            return self._to_result(result_state)
        except AgentDefinitionError as exc:
            logger.warning(f"Agent definition rejected request: {exc}")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="agent_definition_error",
            )
        except LangGraphUnavailable as exc:
            logger.exception("LangGraph V2 runtime unavailable")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="langgraph_unavailable",
            )
        except Exception as exc:
            logger.exception("LangGraph V2 runtime failed")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="runtime_error",
            )

    async def arun(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        started_at = perf_counter()
        async_metadata = {
            "enabled": settings.AGENT_ASYNC_ENABLED,
            "runtime": "langgraph_v2",
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
            definition = self._load_definition(request)
            graph_app = self.graph_app or build_agent_graph(async_mode=True)
            session_id = self._session_id(request)
            session_state = self._load_session(session_id)
            state = self._create_state(
                request=request,
                definition=definition,
                session_id=session_id,
                session_state=session_state,
                extra_metadata={"async_runtime": async_metadata},
            )
            self._inject_session_state(state, session_state)
            async with asyncio.timeout(definition.timeout_seconds):
                result_state = await self._ainvoke_graph(graph_app, state)
            self._save_session(session_id, result_state)
            result = self._to_result(result_state)
            runtime_metadata = result.metadata.get("async_runtime", {})
            runtime_metadata.update(
                {
                    "enabled": True,
                    "runtime": "langgraph_v2",
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "timeout_seconds": definition.timeout_seconds,
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
        except AgentDefinitionError as exc:
            logger.warning(f"Agent definition rejected async request: {exc}")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="agent_definition_error",
                async_metadata={
                    **async_metadata,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "failed": True,
                    "error": str(exc),
                },
            )
        except TimeoutError as exc:
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc) or "agent async runtime timed out",
                error_type="timeout",
                async_metadata={
                    **async_metadata,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "timed_out": True,
                    "failed": True,
                    "error": str(exc) or "agent async runtime timed out",
                },
            )
        except LangGraphUnavailable as exc:
            logger.exception("LangGraph V2 async runtime unavailable")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="langgraph_unavailable",
                async_metadata={
                    **async_metadata,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "failed": True,
                    "error": str(exc),
                },
            )
        except Exception as exc:
            logger.exception("LangGraph V2 async runtime failed")
            return self._failure_result(
                request=request,
                started_at=started_at,
                reason=str(exc),
                error_type="runtime_error",
                async_metadata={
                    **async_metadata,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "failed": True,
                    "error": str(exc),
                },
            )

    async def astream_events(
        self,
        request: AgentRuntimeRequest,
    ) -> AsyncIterator[dict]:
        started_at = perf_counter()
        yield {
            "event": "status",
            "data": {"status": "analyzing", "message": "正在分析问题"},
        }

        if not settings.AGENT_ASYNC_ENABLED:
            result = await asyncio.to_thread(self.run, request)
            yield {"event": "result", "data": {"result": result.model_dump()}}
            return

        try:
            definition = self._load_definition(request)
            graph_app = self.graph_app or build_agent_graph(async_mode=True)
            event_queue: asyncio.Queue[dict] = asyncio.Queue()
            session_id = self._session_id(request)
            session_state = self._load_session(session_id)
            state = self._create_state(
                request=request,
                definition=definition,
                session_id=session_id,
                session_state=session_state,
                extra_metadata={
                    "_agent_stream_answer_enabled": True,
                    "_agent_stream_event_queue": event_queue,
                },
            )
            self._inject_session_state(state, session_state)

            async def run_graph() -> None:
                try:
                    result_state: dict[str, Any] = dict(state)
                    last_status: str | None = "analyzing"
                    async with asyncio.timeout(definition.timeout_seconds):
                        async for update in self._astream_graph(graph_app, state):
                            node_name, update_state = self._extract_stream_state(update)
                            if update_state is not None:
                                result_state = update_state
                            status_event = self._status_from_node(node_name, result_state)
                            if status_event and status_event["status"] != last_status:
                                last_status = status_event["status"]
                                await event_queue.put(
                                    {"event": "status", "data": status_event}
                                )
                    self._save_session(session_id, result_state)
                    result = self._to_result(result_state)
                    result.metadata.setdefault("async_runtime", {})
                    result.metadata["async_runtime"].update(
                        {
                            "enabled": True,
                            "runtime": "langgraph_v2",
                            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                            "timeout_seconds": definition.timeout_seconds,
                            "timed_out": False,
                            "cancelled": False,
                            "failed": False,
                            "error": None,
                        }
                    )
                    await event_queue.put(
                        {"event": "result", "data": {"result": result.model_dump()}}
                    )
                except BaseException as exc:  # noqa: BLE001 - re-raised by consumer
                    await event_queue.put({"event": "_exception", "data": {"error": exc}})

            graph_task = asyncio.create_task(run_graph())
            while True:
                event = await event_queue.get()
                if event.get("event") == "_exception":
                    raise event["data"]["error"]
                yield event
                if event.get("event") == "result":
                    break
            await graph_task
        except asyncio.CancelledError:
            logger.info("LangGraph agent stream cancelled")
            raise
        except Exception:
            logger.exception("LangGraph V2 agent stream failed")
            raise

    def _failure_result(
        self,
        *,
        request: AgentRuntimeRequest,
        started_at: float,
        reason: str,
        error_type: str,
        async_metadata: dict | None = None,
    ) -> AgentRuntimeResult:
        metadata = {
            "runtime": "langgraph_v2",
            "agent_id": request.agent_id or request.metadata.get("agent_id"),
            "runtime_error": {
                "type": error_type,
                "message": reason,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
            "errors": [reason],
        }
        if async_metadata is not None:
            metadata["async_runtime"] = async_metadata
        return AgentRuntimeResult(
            answer=f"Agent V2 执行失败：{reason}",
            action="failed",
            metadata=metadata,
            trace=[],
        )

    def _invoke_graph(self, graph_app, state: AgentState) -> dict:
        try:
            return graph_app.invoke(
                state,
                config={"recursion_limit": self._recursion_limit(state)},
            )
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            return graph_app.invoke(state)

    async def _ainvoke_graph(self, graph_app, state: AgentState) -> dict:
        try:
            return await graph_app.ainvoke(
                state,
                config={"recursion_limit": self._recursion_limit(state)},
            )
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            return await graph_app.ainvoke(state)

    async def _astream_graph(self, graph_app, state: AgentState):
        try:
            async for update in graph_app.astream(
                state,
                config={"recursion_limit": self._recursion_limit(state)},
            ):
                yield update
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            async for update in graph_app.astream(state):
                yield update

    def _load_definition(self, request: AgentRuntimeRequest) -> AgentDefinition:
        return get_agent_definition_registry().get(request.agent_id)

    def _create_state(
        self,
        *,
        request: AgentRuntimeRequest,
        definition: AgentDefinition,
        session_id: str,
        session_state: MemoryState | None,
        extra_metadata: dict | None = None,
    ) -> AgentState:
        knowledge_base_id = (
            request.knowledge_base_id
            if request.knowledge_base_id is not None
            else definition.default_knowledge_base_id
        )
        state = create_initial_state(
            query=request.query,
            conversation_id=request.conversation_id,
            knowledge_base_id=knowledge_base_id,
            memory_context=request.memory_context,
            metadata=self._metadata_with_session(
                {
                    **request.metadata,
                    **(extra_metadata or {}),
                },
                definition=definition,
                session_id=session_id,
                session_state=session_state,
            ),
        )
        state["messages"].insert(
            0,
            {
                "role": "system",
                "content": definition.instructions,
            },
        )
        if request.agent_id is not None:
            state["budget"]["max_steps"] = definition.max_steps
        return state

    def _recursion_limit(self, state: AgentState) -> int:
        budget = state.get("budget") or {}
        return int(budget.get("max_steps") or settings.AGENT_MAX_STEPS) + 8

    def _extract_stream_state(self, update) -> tuple[str | None, dict | None]:
        if not isinstance(update, dict):
            return None, None
        if "metadata" in update or "messages" in update:
            return None, update
        for node_name, node_state in update.items():
            if isinstance(node_state, dict):
                return str(node_name), node_state
        return None, None

    def _status_from_node(self, node_name: str | None, state: dict) -> dict | None:
        if node_name == "planner":
            return {"status": "analyzing", "message": "正在分析问题"}
        if node_name == "tool":
            selected_tools = [
                str(item.get("tool_name") or item.get("name") or "")
                for item in state.get("tool_calls", [])
                if isinstance(item, dict)
            ]
            if "knowledge_search" in selected_tools:
                return {"status": "retrieving", "message": "正在查询知识库"}
            return {"status": "processing", "message": "正在处理任务"}
        if node_name == "observation":
            return {"status": "processing", "message": "正在处理任务结果"}
        if node_name == "reflection":
            return {"status": "processing", "message": "正在调整处理思路"}
        if node_name == "final":
            return {"status": "answering", "message": "正在整理答案"}
        return None

    def _session_id(self, request: AgentRuntimeRequest) -> str:
        metadata_session_id = request.metadata.get("session_id")
        if metadata_session_id:
            return str(metadata_session_id)
        if request.conversation_id is not None:
            return f"conversation:{request.conversation_id}"
        digest = hashlib.sha256(
            f"{request.agent_id or ''}:{request.query}".encode()
        ).hexdigest()[:16]
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
        definition: AgentDefinition,
        session_id: str,
        session_state: MemoryState | None,
    ) -> dict:
        return {
            **metadata,
            "runtime": "langgraph_v2",
            "agent_id": definition.id,
            "agent_definition_version": definition.version,
            "planner_strategy": definition.planner_strategy,
            "max_steps": definition.max_steps,
            "timeout_seconds": definition.timeout_seconds,
            "tool_allowlist": list(definition.tool_allowlist),
            "workflow_allowlist": list(definition.workflow_allowlist),
            "memory_policy": definition.memory_policy,
            "retrieval_policy": definition.retrieval_policy,
            "model_config_keys": sorted(definition.model_settings),
            "default_knowledge_base_id": definition.default_knowledge_base_id,
            "output_mode": definition.output_mode,
            "safety_policy": definition.safety_policy,
            "agent_definition": {
                "id": definition.id,
                "name": definition.name,
                "version": definition.version,
                "planner_strategy": definition.planner_strategy,
                "tool_allowlist": list(definition.tool_allowlist),
                "workflow_allowlist": list(definition.workflow_allowlist),
                "memory_policy": definition.memory_policy,
                "retrieval_policy": definition.retrieval_policy,
                "model_config": definition.model_settings,
                "max_steps": definition.max_steps,
                "timeout_seconds": definition.timeout_seconds,
                "output_mode": definition.output_mode,
                "safety_policy": definition.safety_policy,
            },
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
                    "runtime": "langgraph_v2",
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
        metadata["runtime"] = "langgraph_v2"
        agent_definition = metadata.get("agent_definition")
        if isinstance(agent_definition, dict):
            sanitized_definition = dict(agent_definition)
            model_config = sanitized_definition.pop("model_config", {})
            sanitized_definition["model_config_keys"] = (
                sorted(model_config) if isinstance(model_config, dict) else []
            )
            metadata["agent_definition"] = sanitized_definition
        for key in list(metadata):
            if key.startswith("_agent_stream_"):
                metadata.pop(key, None)
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
