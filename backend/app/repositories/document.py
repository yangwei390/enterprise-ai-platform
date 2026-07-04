from __future__ import annotations

from typing import Any

from sqlalchemy import select

from backend.app.models import Document
from backend.app.repositories.base import BaseRepository


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

    def update(self, document: Document, data: dict[str, Any]) -> Document:
        for field, value in data.items():
            setattr(document, field, value)

        super().update()
        self.db.refresh(document)
        return document

    def delete(self, document: Document) -> None:
        super().delete(document)
