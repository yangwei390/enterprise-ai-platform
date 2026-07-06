from backend.app.logger import logger
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.hybrid.base import HybridRetrieveQuery


class DummySparseRetriever:
    def retrieve(self, query: HybridRetrieveQuery) -> list[RetrievedChunk]:
        logger.info("Sparse retriever is not implemented, returning empty result")
        return []
