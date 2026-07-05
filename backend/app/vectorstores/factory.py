from backend.app.vectorstores.base import BaseVectorStore
from backend.app.vectorstores.qdrant_store import QdrantVectorStore


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(provider: str | None = None) -> BaseVectorStore:
        return QdrantVectorStore()
