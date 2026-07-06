from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.app.models.base import Base
from backend.app.models.mixins import SoftDeleteMixin, TimestampMixin
from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .conversation import Conversation


class Message(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages",
    )
