from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.models.base import Base
from backend.app.models.mixins import SoftDeleteMixin, TimestampMixin
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .message import Message


class Conversation(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    knowledge_base_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="conversation",
    )
