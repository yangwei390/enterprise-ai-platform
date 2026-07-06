from backend.app.conversations import ConversationService
from backend.app.memory.base import MemoryContext
from backend.app.memory.summary_memory import SummaryMemory
from backend.app.memory.token_budget import TokenBudgetManager
from backend.app.memory.window_memory import WindowMemory


class MemoryService:
    def __init__(self, conversation_service: ConversationService) -> None:
        self.window_memory = WindowMemory(conversation_service)
        self.summary_memory = SummaryMemory(conversation_service)
        self.token_budget = TokenBudgetManager()

    def build_memory_context(
        self,
        conversation_id: int,
        max_history_tokens: int = 1500,
        window_size: int = 10,
    ) -> MemoryContext:
        summary = self.summary_memory.get_summary(conversation_id)
        recent_messages = self.window_memory.load(
            conversation_id=conversation_id,
            limit=window_size,
        )
        trimmed_messages = self.token_budget.trim_messages(
            recent_messages,
            max_tokens=max_history_tokens,
        )
        token_budget_used = self.token_budget.estimate_messages_tokens(trimmed_messages)
        return MemoryContext(
            summary=summary,
            recent_messages=trimmed_messages,
            token_budget_used=token_budget_used,
            metadata={
                "window_size": window_size,
                "max_history_tokens": max_history_tokens,
                "loaded_message_count": len(recent_messages),
                "trimmed_message_count": len(trimmed_messages),
            },
        )

    def update_summary(
        self,
        conversation_id: int,
        window_size: int = 10,
    ) -> str:
        messages = self.window_memory.load(
            conversation_id=conversation_id,
            limit=window_size,
        )
        return self.summary_memory.update_summary(conversation_id, messages)
