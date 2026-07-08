from backend.app.config.settings import settings
from pydantic import BaseModel


class ContextCompressionConfig(BaseModel):
    enabled: bool = True
    provider: str = "rule_based"
    max_chars: int = 6000
    max_chunk_chars: int = 1200
    fail_open: bool = True


def get_context_compression_config() -> ContextCompressionConfig:
    return ContextCompressionConfig(
        enabled=settings.CONTEXT_COMPRESSION_ENABLED,
        provider=settings.CONTEXT_COMPRESSION_PROVIDER,
        max_chars=settings.CONTEXT_COMPRESSION_MAX_CHARS,
        max_chunk_chars=settings.CONTEXT_COMPRESSION_MAX_CHUNK_CHARS,
        fail_open=settings.CONTEXT_COMPRESSION_FAIL_OPEN,
    )
