import json

from backend.app.cache import get_redis_client
from backend.app.config.settings import settings
from backend.app.memory.provider import MemoryProvider
from backend.app.memory.snapshot import MemorySnapshot
from backend.app.memory.state import MemoryState


class RedisMemoryProvider(MemoryProvider):
    def __init__(self, redis_client=None, prefix: str | None = None) -> None:
        self.redis = redis_client or get_redis_client()
        self.prefix = prefix or settings.REDIS_PREFIX

    @property
    def name(self) -> str:
        return "redis"

    def save_session(self, state: MemoryState, ttl_seconds: int | None = None) -> None:
        self._set_json(
            self._key("session", state.session_id),
            state.model_dump(),
            ttl_seconds or settings.REDIS_SESSION_TTL_SECONDS,
        )

    def load_session(self, session_id: str) -> MemoryState | None:
        data = self._get_json(self._key("session", session_id))
        return MemoryState.model_validate(data) if data else None

    def delete_session(self, session_id: str) -> None:
        self.redis.delete(self._key("session", session_id))

    def get_cache(self, cache_key: str) -> dict | None:
        data = self._get_json(self._key("cache", cache_key))
        return data if isinstance(data, dict) else None

    def set_cache(
        self,
        cache_key: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        self._set_json(
            self._key("cache", cache_key),
            value,
            ttl_seconds or settings.REDIS_TOOL_CACHE_TTL_SECONDS,
        )

    def delete_cache(self, cache_key: str | None = None) -> None:
        if cache_key is not None:
            self.redis.delete(self._key("cache", cache_key))
            return
        for key in self.redis.scan_iter(self._key("cache", "*")):
            self.redis.delete(key)

    def save_checkpoint(
        self,
        checkpoint_id: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        self._set_json(
            self._key("checkpoint", checkpoint_id),
            value,
            ttl_seconds or settings.REDIS_CHECKPOINT_TTL_SECONDS,
        )

    def load_checkpoint(self, checkpoint_id: str) -> dict | None:
        data = self._get_json(self._key("checkpoint", checkpoint_id))
        return data if isinstance(data, dict) else None

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.redis.delete(self._key("checkpoint", checkpoint_id))

    def list_checkpoints(self) -> list[str]:
        prefix = self._key("checkpoint", "")
        return [
            str(key).replace(prefix, "", 1)
            for key in self.redis.scan_iter(self._key("checkpoint", "*"))
        ]

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            provider=self.name,
            session_count=self._count("session"),
            cache_count=self._count("cache"),
            checkpoint_count=self._count("checkpoint"),
            metadata={"prefix": self.prefix},
        )

    def _set_json(self, key: str, value: dict, ttl_seconds: int) -> None:
        self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    def _get_json(self, key: str) -> dict | None:
        value = self.redis.get(key)
        if not value:
            return None
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None

    def _key(self, namespace: str, key: str) -> str:
        return f"{self.prefix}:{namespace}:{key}"

    def _count(self, namespace: str) -> int:
        return sum(1 for _ in self.redis.scan_iter(self._key(namespace, "*")))
