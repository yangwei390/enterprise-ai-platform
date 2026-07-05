from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.models.base import Base
from backend.app.models.mixins import SoftDeleteMixin, TimestampMixin
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .document import Document


class KnowledgeBase(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    vector_store: Mapped[str] = mapped_column(String(64), default="qdrant", nullable=False)

    documents: Mapped[list[Document]] = relationship(
        "Document",
        back_populates="knowledge_base",
    )
