from backend.app.context.base import BaseContextBuilder
from backend.app.context.builder import BasicContextBuilder


class ContextBuilderFactory:
    @staticmethod
    def get_builder(provider: str | None = None) -> BaseContextBuilder:
        return BasicContextBuilder()
