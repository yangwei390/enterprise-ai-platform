import pickle
from pathlib import Path

from backend.app.config.settings import PROJECT_ROOT
from backend.app.logger import logger
from backend.app.retrievers.sparse.base import SparseDocument
from backend.app.retrievers.sparse.bm25_index import BM25Index


class BM25IndexManager:
    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path or PROJECT_ROOT / "data" / "bm25" / "index.pkl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = BM25Index()

    def load(self) -> BM25Index:
        if not self.index_path.exists():
            self.index = BM25Index()
            return self.index

        try:
            with self.index_path.open("rb") as file:
                loaded_index = pickle.load(file)

            if not isinstance(loaded_index, BM25Index):
                logger.warning("BM25 index file content is invalid, creating empty index")
                self.index = BM25Index()
                return self.index

            self.index = loaded_index
            return self.index
        except Exception:
            logger.exception("Failed to load BM25 index, creating empty index")
            self.index = BM25Index()
            return self.index

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as file:
            pickle.dump(self.index, file)

    def clear(self) -> None:
        self.index.clear()
        if self.index_path.exists():
            self.index_path.unlink()
        self.index = BM25Index()

    def add_documents(self, documents: list[SparseDocument], save: bool = True) -> None:
        self.index.add_documents(documents)
        if save:
            self.save()

    def remove_document(self, document_id: int, save: bool = True) -> None:
        documents = [
            document
            for document in self.index.documents
            if document.document_id != document_id
        ]
        self.rebuild(documents, save=save)

    def list_document_ids(self) -> list[int]:
        document_ids = {
            document.document_id
            for document in self.index.documents
            if document.document_id is not None
        }
        return sorted(document_ids)

    def rebuild(self, documents: list[SparseDocument], save: bool = True) -> None:
        self.index = BM25Index()
        self.index.add_documents(documents)
        if save:
            self.save()

    def get_index(self) -> BM25Index:
        return self.index


_bm25_index_manager: BM25IndexManager | None = None


def get_bm25_index_manager() -> BM25IndexManager:
    global _bm25_index_manager
    if _bm25_index_manager is None:
        _bm25_index_manager = BM25IndexManager()
        _bm25_index_manager.load()
    return _bm25_index_manager
