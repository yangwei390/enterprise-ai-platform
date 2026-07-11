from dataclasses import dataclass

from backend.app.db.session import SessionLocal
from backend.app.exceptions import BusinessException
from backend.app.models import Document
from backend.app.retrievers.sparse import get_bm25_index_manager
from backend.app.vectorstores import QdrantVectorStore
from sqlalchemy import select


@dataclass
class OrphanDetectionResult:
    postgres_document_ids: list[int]
    qdrant_document_ids: list[int]
    bm25_document_ids: list[int]
    qdrant_orphan_document_ids: list[int]
    bm25_orphan_document_ids: list[int]
    orphan_document_ids: list[int]

    def to_dict(self) -> dict:
        return {
            "postgres_document_ids": self.postgres_document_ids,
            "qdrant_document_ids": self.qdrant_document_ids,
            "bm25_document_ids": self.bm25_document_ids,
            "qdrant_orphan_document_ids": self.qdrant_orphan_document_ids,
            "bm25_orphan_document_ids": self.bm25_orphan_document_ids,
            "orphan_document_ids": self.orphan_document_ids,
        }


class IndexConsistencyService:
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()

    def detect_orphans(self) -> dict:
        postgres_ids = self._postgres_active_document_ids()
        qdrant_ids = self.vector_store.list_document_ids()
        bm25_ids = get_bm25_index_manager().list_document_ids()
        postgres_id_set = set(postgres_ids)
        qdrant_orphans = sorted(set(qdrant_ids) - postgres_id_set)
        bm25_orphans = sorted(set(bm25_ids) - postgres_id_set)
        orphan_ids = sorted(set(qdrant_orphans) | set(bm25_orphans))
        return OrphanDetectionResult(
            postgres_document_ids=postgres_ids,
            qdrant_document_ids=qdrant_ids,
            bm25_document_ids=bm25_ids,
            qdrant_orphan_document_ids=qdrant_orphans,
            bm25_orphan_document_ids=bm25_orphans,
            orphan_document_ids=orphan_ids,
        ).to_dict()

    def cleanup_orphan(self, document_id: int) -> dict:
        detection = self.detect_orphans()
        orphan_ids = set(detection["orphan_document_ids"])
        if document_id not in orphan_ids:
            raise BusinessException(40402, "孤儿索引不存在")
        qdrant_result = self.vector_store.delete_by_document_id(document_id)
        get_bm25_index_manager().remove_document(document_id, save=True)
        return {
            "document_id": document_id,
            "qdrant": qdrant_result,
            "bm25_removed": True,
        }

    def cleanup_orphans(self) -> dict:
        detection = self.detect_orphans()
        results = [
            self.cleanup_orphan(document_id)
            for document_id in detection["orphan_document_ids"]
        ]
        return {
            "before": detection,
            "cleaned_count": len(results),
            "results": results,
            "after": self.detect_orphans(),
        }

    def _postgres_active_document_ids(self) -> list[int]:
        db = SessionLocal()
        try:
            result = db.execute(
                select(Document.id).where(Document.deleted_at.is_(None))
            )
            return sorted(int(document_id) for document_id in result.scalars().all())
        finally:
            db.close()
