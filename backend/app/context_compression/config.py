from backend.app.config.settings import settings
from pydantic import BaseModel


class ContextCompressionConfig(BaseModel):
    enabled: bool = True
    provider: str = "rule_based"
    max_chars: int = 6000
    max_chunk_chars: int = 1200
    fail_open: bool = True
    llm_model: str = "qwen-turbo"
    llm_temperature: float = 0
    llm_max_chars_per_chunk: int = 1200
    llm_timeout_seconds: int = 30
    llm_max_calls: int = 8


def get_context_compression_config() -> ContextCompressionConfig:
    return ContextCompressionConfig(
        enabled=settings.CONTEXT_COMPRESSION_ENABLED,
        provider=settings.CONTEXT_COMPRESSION_PROVIDER,
        max_chars=settings.CONTEXT_COMPRESSION_MAX_CHARS,
        max_chunk_chars=settings.CONTEXT_COMPRESSION_MAX_CHUNK_CHARS,
        fail_open=settings.CONTEXT_COMPRESSION_FAIL_OPEN,
        llm_model=settings.CONTEXT_COMPRESSION_LLM_MODEL,
        llm_temperature=settings.CONTEXT_COMPRESSION_LLM_TEMPERATURE,
        llm_max_chars_per_chunk=settings.CONTEXT_COMPRESSION_LLM_MAX_CHARS_PER_CHUNK,
        llm_timeout_seconds=settings.CONTEXT_COMPRESSION_LLM_TIMEOUT_SECONDS,
        llm_max_calls=settings.CONTEXT_COMPRESSION_LLM_MAX_CALLS,
    )
