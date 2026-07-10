from backend.app.memory.providers.memory_provider import InMemoryMemoryProvider


class SemanticMemoryProvider(InMemoryMemoryProvider):
    @property
    def name(self) -> str:
        return "semantic"
