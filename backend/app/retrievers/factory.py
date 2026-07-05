from backend.app.retrievers.base import BaseRetriever
from backend.app.retrievers.qdrant_retriever import QdrantRetriever


class RetrieverFactory:
    @staticmethod
    def get_retriever(provider: str | None = None) -> BaseRetriever:
        return QdrantRetriever()
