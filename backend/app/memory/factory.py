from functools import lru_cache

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.memory.manager import CheckpointManager, MemoryManager
from backend.app.memory.providers import InMemoryMemoryProvider, RedisMemoryProvider


class MemoryFactory:
    @staticmethod
    def get_manager(provider: str | None = None) -> MemoryManager:
        selected_provider = (provider or settings.MEMORY_PROVIDER).lower()
        if selected_provider == "redis" and settings.REDIS_ENABLED:
            try:
                return MemoryManager(RedisMemoryProvider())
            except Exception as exc:
                logger.warning(f"Redis memory unavailable, fallback to memory: {exc}")
        return MemoryManager(_get_in_memory_provider())

    @staticmethod
    def get_checkpoint_manager(provider: str | None = None) -> CheckpointManager:
        return CheckpointManager(MemoryFactory.get_manager(provider))


@lru_cache
def _get_in_memory_provider() -> InMemoryMemoryProvider:
    return InMemoryMemoryProvider()
