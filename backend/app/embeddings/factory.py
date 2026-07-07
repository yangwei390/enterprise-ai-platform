from backend.app.embeddings.base import BaseEmbedding
from backend.app.embeddings.config import get_embedding_config
from backend.app.embeddings.dashscope_embedding import DashScopeEmbedding
from backend.app.embeddings.dummy_embedding import DummyEmbedding


class EmbeddingFactory:
    @staticmethod
    def get_embedding(provider: str | None = None) -> BaseEmbedding:
        config = get_embedding_config()
        selected_provider = (provider or config.provider).lower()
        if selected_provider == "dashscope":
            return DashScopeEmbedding(config=config)
        return DummyEmbedding()
