from backend.app.retrievers.hybrid.base import HybridRetrieveQuery, HybridRetrieveResult
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.hybrid.fusion import rrf_fusion
from backend.app.retrievers.hybrid.sparse_retriever import BM25SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()

    def retrieve(self, query: HybridRetrieveQuery) -> HybridRetrieveResult:
        dense_chunks = self.dense_retriever.retrieve(query)
        sparse_chunks = self.sparse_retriever.retrieve(query)
        fused_chunks = rrf_fusion(
            dense_chunks=dense_chunks,
            sparse_chunks=sparse_chunks,
            top_k=query.top_k,
        )

        return HybridRetrieveResult(
            chunks=fused_chunks,
            total=len(fused_chunks),
            metadata={
                "retriever_mode": "hybrid",
                "dense_total": len(dense_chunks),
                "sparse_total": len(sparse_chunks),
                "fused_total": len(fused_chunks),
                "fusion": "rrf",
                "bm25_enabled": True,
            },
        )
