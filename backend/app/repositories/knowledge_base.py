from datetime import datetime
from typing import Any

from backend.app.models import Document, KnowledgeBase
from backend.app.repositories.base import BaseRepository
from sqlalchemy import select


class KnowledgeBaseRepository(BaseRepository):
    def create(self, data: dict[str, Any]) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(**data)
        return super().create(knowledge_base)

    def get(self, id: int) -> KnowledgeBase | None:
        result = self.db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == id,
                KnowledgeBase.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    def list(self) -> list[KnowledgeBase]:
        result = self.db.execute(
            select(KnowledgeBase).where(KnowledgeBase.deleted_at.is_(None))
        )
        return list(result.scalars().all())

    def update(self, knowledge_base: KnowledgeBase, data: dict[str, Any]) -> KnowledgeBase:
        for field, value in data.items():
            setattr(knowledge_base, field, value)

        super().update()
        self.db.refresh(knowledge_base)
        return knowledge_base

    def has_active_documents(self, knowledge_base_id: int) -> bool:
        result = self.db.execute(
            select(Document.id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.deleted_at.is_(None),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    def delete(self, knowledge_base: KnowledgeBase) -> None:
        knowledge_base.deleted_at = datetime.utcnow()
        self.db.commit()
