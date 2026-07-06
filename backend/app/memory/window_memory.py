from backend.app.conversations import ConversationService
from backend.app.memory.base import MemoryMessage


class WindowMemory:
    def __init__(self, conversation_service: ConversationService) -> None:
        self.conversation_service = conversation_service

    def load(self, conversation_id: int, limit: int = 10) -> list[MemoryMessage]:
        messages = self.conversation_service.get_recent_messages(
            conversation_id=conversation_id,
            limit=limit,
        )
        return [
            MemoryMessage(
                role=message.role,
                content=message.content,
                metadata=message.message_metadata or {},
            )
            for message in messages
            if message.role != "system"
        ]
