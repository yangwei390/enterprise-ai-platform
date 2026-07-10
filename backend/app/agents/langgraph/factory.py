from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.runtime import AgentRuntime
from backend.app.config.settings import settings
from backend.app.logger import logger


class AgentRuntimeFactory:
    @staticmethod
    def get_runtime(provider: str | None = None):
        selected_provider = (provider or settings.AGENT_RUNTIME).lower()
        if selected_provider == "v1":
            return AgentRuntime()
        if selected_provider == "langgraph":
            return LangGraphAgentRuntime()

        logger.warning(
            "Unknown agent runtime provider, fallback to v1: {}",
            selected_provider,
        )
        return AgentRuntime()
