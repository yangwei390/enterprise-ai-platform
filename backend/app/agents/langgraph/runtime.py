from backend.app.agents.langgraph.graph import LangGraphUnavailable, build_agent_graph
from backend.app.agents.langgraph.state import create_initial_state
from backend.app.agents.runtime import AgentRuntime
from backend.app.agents.state import AgentRuntimeRequest, AgentRuntimeResult
from backend.app.agents.trace import AgentTraceStep
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
