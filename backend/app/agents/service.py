from backend.app.agents.base import AgentDefinition, AgentRunRequest, AgentRunResult
from backend.app.agents.executor import AgentExecutor


class AgentService:
    def run(self, request: AgentRunRequest) -> AgentRunResult:
        definition = AgentDefinition(
            id=request.agent_id or "default-agent",
            name="Default Agent",
            max_steps=8,
            enable_reflection=True,
        )
        return AgentExecutor(definition).run(request)
