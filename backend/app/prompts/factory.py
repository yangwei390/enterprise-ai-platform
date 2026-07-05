from backend.app.prompts.base import BasePromptBuilder
from backend.app.prompts.builder import BasicPromptBuilder


class PromptBuilderFactory:
    @staticmethod
    def get_builder(provider: str | None = None) -> BasePromptBuilder:
        return BasicPromptBuilder()
