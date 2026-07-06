from datetime import datetime

from backend.app.conversations.repository import ConversationRepository
from backend.app.exceptions import BusinessException
from backend.app.models import Conversation, Message
from backend.app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationService:
    def __init__(self, repository: ConversationRepository) -> None:
        self.repository = repository

    def create_conversation(self, data: ConversationCreate) -> Conversation:
        return self.repository.create_conversation(
            title=data.title,
            knowledge_base_id=data.knowledge_base_id,
        )

    def list_conversations(self) -> list[Conversation]:
        return self.repository.list_conversations()

    def get_conversation(self, id: int) -> Conversation:
        conversation = self.repository.get_conversation(id)
        if conversation is None:
            raise BusinessException(40403, "会话不存在")
        return conversation

    def update_conversation(
        self,
        id: int,
        data: ConversationUpdate,
    ) -> Conversation:
        conversation = self.get_conversation(id)
        update_data = data.model_dump(exclude_unset=True)
        return self.repository.update_conversation(conversation, update_data)

    def delete_conversation(self, id: int) -> None:
        conversation = self.get_conversation(id)
        self.repository.soft_delete_conversation(conversation)

    def add_user_message(
        self,
        conversation_id: int,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        self.get_conversation(conversation_id)
        return self.repository.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        conversation_id: int,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        self.get_conversation(conversation_id)
        return self.repository.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )

    def get_recent_messages(self, conversation_id: int, limit: int = 10) -> list[Message]:
        self.get_conversation(conversation_id)
        return self.repository.list_messages(conversation_id, limit=limit)

    def list_messages(self, conversation_id: int) -> list[Message]:
        self.get_conversation(conversation_id)
        return self.repository.list_messages(conversation_id)

    def update_summary(self, conversation_id: int, summary: str) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        return self.repository.update_conversation_summary(
            conversation=conversation,
            summary=summary,
            summary_updated_at=datetime.utcnow(),
        )
