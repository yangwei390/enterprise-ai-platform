from backend.app.memory.providers.memory_provider import InMemoryMemoryProvider


class ConversationMemoryProvider(InMemoryMemoryProvider):
    @property
    def name(self) -> str:
        return "conversation"
