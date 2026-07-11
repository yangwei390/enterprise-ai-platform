import hashlib
import json

from backend.app.config.settings import settings
from backend.app.indexing import IndexVersionManager
from backend.app.memory.provider import MemoryProvider
from backend.app.memory.snapshot import MemorySnapshot
from backend.app.memory.state import MemoryState


class MemoryManager:
    def __init__(self, provider: MemoryProvider) -> None:
        self.provider = provider

    def save_session(self, state: MemoryState) -> None:
        self.provider.save_session(state, settings.REDIS_SESSION_TTL_SECONDS)

    def load_session(self, session_id: str) -> MemoryState | None:
        return self.provider.load_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.provider.delete_session(session_id)

    def get_tool_cache(self, tool_name: str, arguments: dict) -> tuple[dict | None, str]:
        cache_key = self.build_tool_cache_key(tool_name, arguments)
        return self.provider.get_cache(cache_key), cache_key

    def set_tool_cache(
        self,
        tool_name: str,
        arguments: dict,
        result: dict,
    ) -> str:
        cache_key = self.build_tool_cache_key(tool_name, arguments)
        self.provider.set_cache(
            cache_key,
            result,
            settings.REDIS_TOOL_CACHE_TTL_SECONDS,
        )
        return cache_key

    def delete_cache(self, cache_key: str | None = None) -> None:
        self.provider.delete_cache(cache_key)

    def save_checkpoint(self, checkpoint_id: str, value: dict) -> None:
        self.provider.save_checkpoint(
            checkpoint_id,
            value,
            settings.REDIS_CHECKPOINT_TTL_SECONDS,
        )

    def load_checkpoint(self, checkpoint_id: str) -> dict | None:
        return self.provider.load_checkpoint(checkpoint_id)

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.provider.delete_checkpoint(checkpoint_id)

    def list_checkpoints(self) -> list[str]:
        return self.provider.list_checkpoints()

    def snapshot(self) -> MemorySnapshot:
        return self.provider.snapshot()

    def build_tool_cache_key(self, tool_name: str, arguments: dict) -> str:
        cache_arguments = dict(arguments)
        if tool_name == "knowledge_search":
            knowledge_base_id = cache_arguments.get("knowledge_base_id")
            version = IndexVersionManager().get_version(
                knowledge_base_id if isinstance(knowledge_base_id, int) else None
            )
            cache_arguments["knowledge_base_index_version"] = version
        raw = json.dumps(cache_arguments, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{tool_name}:{digest}"


class CheckpointManager:
    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    def save(self, checkpoint_id: str, value: dict) -> None:
        self.memory_manager.save_checkpoint(checkpoint_id, value)

    def load(self, checkpoint_id: str) -> dict | None:
        return self.memory_manager.load_checkpoint(checkpoint_id)

    def delete(self, checkpoint_id: str) -> None:
        self.memory_manager.delete_checkpoint(checkpoint_id)

    def list(self) -> list[str]:
        return self.memory_manager.list_checkpoints()
