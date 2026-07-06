from backend.app.retrievers.sparse.base import (
    SparseDocument,
    SparseSearchQuery,
    SparseSearchResult,
)
from backend.app.retrievers.sparse.bm25_index import BM25Index


class BM25Retriever:
    def __init__(self, index: BM25Index | None = None) -> None:
        self.index = index or BM25Index()

    def add_documents(self, documents: list[SparseDocument]) -> None:
        self.index.add_documents(documents)

    def retrieve(self, query: SparseSearchQuery) -> list[SparseSearchResult]:
        return self.index.search(query)
