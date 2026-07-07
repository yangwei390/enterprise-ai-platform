from backend.app.logger import logger
from backend.app.rerankers.base import BaseReranker
from backend.app.rerankers.config import get_reranker_config
from backend.app.rerankers.dashscope_reranker import DashScopeReranker
from backend.app.rerankers.dummy_reranker import DummyReranker


class RerankerFactory:
    @staticmethod
    def get_reranker(provider: str | None = None) -> BaseReranker:
        config = get_reranker_config()
        selected_provider = (provider or config.provider).lower()
        if selected_provider == "dummy":
            return DummyReranker()
        if selected_provider == "dashscope":
            return DashScopeReranker(config=config)

        logger.warning(f"Unknown reranker provider: {selected_provider}, fallback to dummy")
        return DummyReranker()
