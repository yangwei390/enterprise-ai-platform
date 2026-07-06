from backend.app.chunkers import Chunk
from backend.app.logger import logger
from backend.app.models import Document
from backend.app.retrievers.sparse import SparseDocument, get_bm25_index_manager


class DocumentIndexSynchronizer:
    def sync_bm25_for_document(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> dict:
        try:
            manager = get_bm25_index_manager()
            sparse_documents = [
                SparseDocument(
                    id=self._get_chunk_id(chunk, document.id),
                    text=chunk.text,
                    document_id=document.id,
                    knowledge_base_id=document.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    metadata=dict(chunk.metadata),
                )
                for chunk in chunks
            ]

            manager.remove_document(document.id, save=True)
            manager.add_documents(sparse_documents, save=True)

            return {
                "bm25_indexed": True,
                "bm25_indexed_count": len(sparse_documents),
                "bm25_replaced": True,
                "bm25_index_path": str(manager.index_path),
            }
        except Exception as exc:
            logger.exception("BM25 document index sync failed")
            return {
                "bm25_indexed": False,
                "bm25_replaced": False,
                "bm25_indexed_count": 0,
                "bm25_error": str(exc),
            }

    def _get_chunk_id(self, chunk: Chunk, document_id: int | None) -> str:
        chunk_id = getattr(chunk, "id", None)
        if chunk_id:
            return str(chunk_id)
        return f"{document_id}_{chunk.chunk_index}"
