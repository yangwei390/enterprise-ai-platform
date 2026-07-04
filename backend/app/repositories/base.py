from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session


ModelType = TypeVar("ModelType")


class BaseRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, model: type[ModelType], id: int) -> ModelType | None:
        return self.db.get(model, id)

    def list_all(self, model: type[ModelType]) -> list[ModelType]:
        result = self.db.execute(select(model))
        return list(result.scalars().all())

    def create(self, obj: ModelType) -> ModelType:
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def update(self) -> None:
        self.db.commit()

    def delete(self, obj: Any) -> None:
        self.db.delete(obj)
        self.db.commit()
