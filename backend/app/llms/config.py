from backend.app.config.settings import settings
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "dummy"
    model: str = "dummy-llm"
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout: int = 30
    base_url: str | None = None
    api_key: str | None = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        stream=settings.LLM_STREAM,
    )
