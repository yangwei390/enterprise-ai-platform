from typing import Any

from backend.app.models import KnowledgeBase
from backend.app.repositories.base import BaseRepository


class KnowledgeBaseRepository(BaseRepository):
    def create(self, data: dict[str, Any]) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(**data)
        return super().create(knowledge_base)

    def get(self, id: int) -> KnowledgeBase | None:
        return self.get_by_id(KnowledgeBase, id)

    def list(self) -> list[KnowledgeBase]:
        return self.list_all(KnowledgeBase)

    def update(self, knowledge_base: KnowledgeBase, data: dict[str, Any]) -> KnowledgeBase:
        for field, value in data.items():
            setattr(knowledge_base, field, value)

        super().update()
        self.db.refresh(knowledge_base)
        return knowledge_base

    def delete(self, knowledge_base: KnowledgeBase) -> None:
        super().delete(knowledge_base)
