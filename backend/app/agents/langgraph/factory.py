from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.logger import logger


class AgentRuntimeFactory:
    @staticmethod
    def get_runtime(provider: str | None = None):
        selected_provider = (provider or "langgraph").lower()
        if selected_provider not in {"langgraph", "v2", "langgraph_v2"}:
            logger.warning(
                "Unknown agent runtime provider, using LangGraph V2: {}",
                selected_provider,
            )
        return LangGraphAgentRuntime()
