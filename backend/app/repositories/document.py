from __future__ import annotations

from typing import Any

from backend.app.models import Document, KnowledgeBase
from backend.app.repositories.base import BaseRepository
from sqlalchemy import select


class DocumentRepository(BaseRepository):
    def create(self, data: dict[str, Any]) -> Document:
        document = Document(**data)
        return super().create(document)

    def get(self, id: int) -> Document | None:
        return self.get_by_id(Document, id)

    def list(self) -> list[Document]:
        return self.list_all(Document)

    def list_by_knowledge_base_id(self, knowledge_base_id: int) -> list[Document]:
        result = self.db.execute(
            select(Document).where(Document.knowledge_base_id == knowledge_base_id)
        )
        return list(result.scalars().all())

    def list_active_by_knowledge_base_id(self, knowledge_base_id: int) -> list[Document]:
        result = self.db.execute(
            select(Document).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.deleted_at.is_(None),
            )
        )
        return list(result.scalars().all())

    def knowledge_base_exists(self, knowledge_base_id: int) -> bool:
        result = self.db.execute(
            select(KnowledgeBase.id)
            .where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.deleted_at.is_(None),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    def update(self, document: Document, data: dict[str, Any]) -> Document:
        for field, value in data.items():
            setattr(document, field, value)

        super().update()
        self.db.refresh(document)
        return document

    def delete(self, document: Document) -> None:
        super().delete(document)
