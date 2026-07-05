from backend.app.llms.base import BaseLLM
from backend.app.llms.config import get_llm_config
from backend.app.llms.providers import DummyLLMProvider, OpenAIProvider
from backend.app.llms.providers.dashscope_provider import DashScopeProvider
from backend.app.logger import logger


class LLMFactory:
    @staticmethod
    def get_llm(provider: str | None = None) -> BaseLLM:
        config = get_llm_config()
        selected_provider = provider or config.provider

        if selected_provider == "dummy":
            return DummyLLMProvider(config=config)

        if selected_provider == "openai":
            return OpenAIProvider(config=config)

        if selected_provider == "dashscope":
            return DashScopeProvider(config=config)

        logger.warning(
            f"LLM provider not implemented, fallback to dummy: {selected_provider}"
        )
        return DummyLLMProvider(config=config)
