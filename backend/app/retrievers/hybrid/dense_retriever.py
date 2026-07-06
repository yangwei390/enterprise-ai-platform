from backend.app.retrievers.base import RetrievedChunk, RetrieveQuery
from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.qdrant_retriever import QdrantRetriever


class DenseRetriever:
    def __init__(self, retriever: QdrantRetriever | None = None) -> None:
        self.retriever = retriever or QdrantRetriever()

    def retrieve(self, query: HybridRetrieveQuery) -> list[RetrievedChunk]:
        result = self.retriever.retrieve(
            RetrieveQuery(
                query=query.query,
                knowledge_base_id=query.knowledge_base_id,
                top_k=query.top_k,
                score_threshold=query.score_threshold,
                metadata_filter=query.metadata_filter,
            )
        )
        return result.chunks
