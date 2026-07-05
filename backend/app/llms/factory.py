from backend.app.llms.base import BaseLLM
from backend.app.llms.dummy_llm import DummyLLM


class LLMFactory:
    @staticmethod
    def get_llm(provider: str | None = None) -> BaseLLM:
        return DummyLLM()
