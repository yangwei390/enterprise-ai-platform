from backend.app.rerankers.base import BaseReranker
from backend.app.rerankers.dummy_reranker import DummyReranker


class RerankerFactory:
    @staticmethod
    def get_reranker(provider: str | None = None) -> BaseReranker:
        return DummyReranker()
