import time

from backend.app.memory.provider import MemoryProvider
from backend.app.memory.snapshot import MemorySnapshot
from backend.app.memory.state import MemoryState


class InMemoryMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self.sessions: dict[str, tuple[dict, float | None]] = {}
        self.cache: dict[str, tuple[dict, float | None]] = {}
        self.checkpoints: dict[str, tuple[dict, float | None]] = {}

    @property
    def name(self) -> str:
        return "memory"

    def save_session(self, state: MemoryState, ttl_seconds: int | None = None) -> None:
        self.sessions[state.session_id] = (state.model_dump(), _expires_at(ttl_seconds))

    def load_session(self, session_id: str) -> MemoryState | None:
        value = self._get(self.sessions, session_id)
        return MemoryState.model_validate(value) if value else None

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def get_cache(self, cache_key: str) -> dict | None:
        return self._get(self.cache, cache_key)

    def set_cache(
        self,
        cache_key: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        self.cache[cache_key] = (value, _expires_at(ttl_seconds))

    def delete_cache(self, cache_key: str | None = None) -> None:
        if cache_key is None:
            self.cache.clear()
            return
        self.cache.pop(cache_key, None)

    def save_checkpoint(
        self,
        checkpoint_id: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        self.checkpoints[checkpoint_id] = (value, _expires_at(ttl_seconds))

    def load_checkpoint(self, checkpoint_id: str) -> dict | None:
        return self._get(self.checkpoints, checkpoint_id)

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.checkpoints.pop(checkpoint_id, None)

    def list_checkpoints(self) -> list[str]:
        self._purge_expired(self.checkpoints)
        return list(self.checkpoints)

    def snapshot(self) -> MemorySnapshot:
        self._purge_expired(self.sessions)
        self._purge_expired(self.cache)
        self._purge_expired(self.checkpoints)
        return MemorySnapshot(
            provider=self.name,
            session_count=len(self.sessions),
            cache_count=len(self.cache),
            checkpoint_count=len(self.checkpoints),
        )

    def _get(self, store: dict[str, tuple[dict, float | None]], key: str) -> dict | None:
        item = store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and expires_at < time.time():
            store.pop(key, None)
            return None
        return value

    def _purge_expired(self, store: dict[str, tuple[dict, float | None]]) -> None:
        for key in list(store):
            self._get(store, key)


def _expires_at(ttl_seconds: int | None) -> float | None:
    return time.time() + ttl_seconds if ttl_seconds else None
