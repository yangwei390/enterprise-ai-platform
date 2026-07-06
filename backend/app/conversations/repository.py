from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.app.models import Conversation, Message
from sqlalchemy import select
from sqlalchemy.orm import Session


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_conversation(
        self,
        title: str | None,
        knowledge_base_id: int | None,
    ) -> Conversation:
        conversation = Conversation(title=title, knowledge_base_id=knowledge_base_id)
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_conversation(self, id: int) -> Conversation | None:
        result = self.db.execute(
            select(Conversation).where(
                Conversation.id == id,
                Conversation.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    def list_conversations(self) -> list[Conversation]:
        result = self.db.execute(
            select(Conversation)
            .where(Conversation.deleted_at.is_(None))
            .order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    def update_conversation(
        self,
        conversation: Conversation,
        data: dict[str, Any],
    ) -> Conversation:
        for field, value in data.items():
            setattr(conversation, field, value)

        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_conversation_summary(
        self,
        conversation: Conversation,
        summary: str,
        summary_updated_at: datetime,
    ) -> Conversation:
        conversation.summary = summary
        conversation.summary_updated_at = summary_updated_at
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def soft_delete_conversation(self, conversation: Conversation) -> None:
        conversation.deleted_at = datetime.utcnow()
        self.db.commit()

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def list_messages(
        self,
        conversation_id: int,
        limit: int | None = None,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.created_at.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)

        result = self.db.execute(statement)
        messages = list(result.scalars().all())
        return list(reversed(messages))
