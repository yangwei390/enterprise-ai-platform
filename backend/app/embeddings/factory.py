from backend.app.embeddings.base import BaseEmbedding
from backend.app.embeddings.dummy_embedding import DummyEmbedding


class EmbeddingFactory:
    @staticmethod
    def get_embedding(provider: str | None = None) -> BaseEmbedding:
        return DummyEmbedding()
