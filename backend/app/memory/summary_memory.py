from backend.app.conversations import ConversationService
from backend.app.memory.base import MemoryMessage


class SummaryMemory:
    def __init__(self, conversation_service: ConversationService) -> None:
        self.conversation_service = conversation_service

    def get_summary(self, conversation_id: int) -> str | None:
        conversation = self.conversation_service.get_conversation(conversation_id)
        return conversation.summary

    def update_summary(
        self,
        conversation_id: int,
        messages: list[MemoryMessage],
    ) -> str:
        summary = self._build_simple_summary(messages)
        self.conversation_service.update_summary(conversation_id, summary)
        return summary

    def _build_simple_summary(self, messages: list[MemoryMessage]) -> str:
        lines = []
        for message in messages:
            role = _format_role(message.role)
            lines.append(f"{role}：{message.content}")
        return "\n".join(lines)[:1000]


def _format_role(role: str) -> str:
    if role == "user":
        return "用户"
    if role == "assistant":
        return "助手"
    return role
