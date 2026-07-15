from backend.app.agents.base import AgentRunRequest, AgentRunResult, AgentRunStatus, AgentStep
from backend.app.agents.factory import AgentRuntimeFactory
from backend.app.agents.state import AgentRuntimeRequest


class AgentService:
    def run(self, request: AgentRunRequest) -> AgentRunResult:
        runtime_request = AgentRuntimeRequest(
            query=request.task,
            agent_id=request.agent_id,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=request.conversation_id,
            memory_context=None,
            metadata={
                **request.metadata,
                "enable_tools": request.enable_tools,
                "enable_memory": request.enable_memory,
            },
        )
        result = AgentRuntimeFactory.get_runtime().run(runtime_request)
        failed = result.action == "failed"
        runtime_error = result.metadata.get("runtime_error", {}).get("message")
        return AgentRunResult(
            task=request.task,
            status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCESS,
            answer=result.answer,
            steps=[
                AgentStep(
                    index=0,
                    type="runtime",
                    name="langgraph_v2_runtime",
                    input={"query": request.task},
                    output=result.model_dump(),
                    status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCESS,
                    error=runtime_error if failed else None,
                )
            ],
            artifacts=[],
            metadata={
                "runtime": "langgraph_v2",
                "agent_runtime": result.metadata,
                "sources": result.sources,
                "citations": result.citations,
            },
            error=runtime_error if failed else None,
        )
