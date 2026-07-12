from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.hybrid.base import HybridRetrieveQuery
from backend.app.retrievers.sparse import BM25Retriever, SparseSearchQuery, get_bm25_index_manager


class BM25SparseRetriever:
    def __init__(self, retriever: BM25Retriever | None = None) -> None:
        manager = get_bm25_index_manager()
        self.retriever = retriever or BM25Retriever(index=manager.get_index())

    def retrieve(self, query: HybridRetrieveQuery) -> list[RetrievedChunk]:
        results = self.retriever.retrieve(
            SparseSearchQuery(
                query=query.query,
                knowledge_base_id=query.knowledge_base_id,
                top_k=query.top_k,
                metadata_filter=query.metadata_filter,
                constraints=query.constraints,
            )
        )
        return [
            RetrievedChunk(
                id=result.id,
                score=result.score,
                text=result.text,
                document_id=result.document_id,
                knowledge_base_id=result.knowledge_base_id,
                chunk_index=result.chunk_index,
                metadata={
                    **result.metadata,
                    "retriever": "bm25",
                    "sparse_score": result.score,
                },
            )
            for result in results
        ]
