from abc import ABC, abstractmethod

from backend.app.memory.snapshot import MemorySnapshot
from backend.app.memory.state import MemoryState


class MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_session(self, state: MemoryState, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_session(self, session_id: str) -> MemoryState | None:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_cache(self, cache_key: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def set_cache(
        self,
        cache_key: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_cache(self, cache_key: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_id: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_checkpoints(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> MemorySnapshot:
        raise NotImplementedError
