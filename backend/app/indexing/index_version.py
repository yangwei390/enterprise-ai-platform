from backend.app.cache import get_redis_client
from backend.app.config.settings import settings
from backend.app.logger import logger

_memory_versions: dict[int, int] = {}


class IndexVersionManager:
    namespace = "kb_index_version"

    def get_version(self, knowledge_base_id: int | None) -> int:
        if knowledge_base_id is None:
            return 0
        try:
            value = get_redis_client().get(self._key(knowledge_base_id))
            return int(value) if value is not None else self._memory_version(knowledge_base_id)
        except Exception as exc:
            logger.warning(
                f"Knowledge base index version Redis get failed | "
                f"knowledge_base_id={knowledge_base_id} | error={exc}"
            )
            return self._memory_version(knowledge_base_id)

    def bump_version(self, knowledge_base_id: int | None) -> int:
        if knowledge_base_id is None:
            return 0
        _memory_versions[knowledge_base_id] = self._memory_version(knowledge_base_id) + 1
        try:
            value = get_redis_client().incr(self._key(knowledge_base_id))
            version = int(value)
            _memory_versions[knowledge_base_id] = max(
                _memory_versions[knowledge_base_id],
                version,
            )
            return version
        except Exception as exc:
            logger.warning(
                f"Knowledge base index version Redis bump failed | "
                f"knowledge_base_id={knowledge_base_id} | error={exc}"
            )
            return _memory_versions[knowledge_base_id]

    def _memory_version(self, knowledge_base_id: int) -> int:
        return _memory_versions.get(knowledge_base_id, 0)

    def _key(self, knowledge_base_id: int) -> str:
        return f"{settings.REDIS_PREFIX}:{self.namespace}:{knowledge_base_id}"
