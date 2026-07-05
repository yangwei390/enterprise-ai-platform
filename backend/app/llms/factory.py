from backend.app.llms.base import BaseLLM
from backend.app.llms.providers import DummyLLMProvider
from backend.app.logger import logger


class LLMFactory:
    @staticmethod
    def get_llm(provider: str | None = None) -> BaseLLM:
        if provider in (None, "", "dummy"):
            return DummyLLMProvider()

        logger.warning(f"LLM provider not implemented, fallback to dummy: {provider}")
        return DummyLLMProvider()
