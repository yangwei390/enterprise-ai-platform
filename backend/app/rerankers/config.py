import os
from functools import lru_cache

from backend.app.config.settings import settings
from pydantic import BaseModel


class RerankerConfig(BaseModel):
    provider: str = "dummy"
    model: str = "gte-rerank-v2"
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    top_k: int = 20
    timeout: int = 30
    fail_open: bool = True


@lru_cache
def get_reranker_config() -> RerankerConfig:
    api_key = (
        settings.RERANK_API_KEY
        or settings.LLM_API_KEY
        or os.getenv("DASHSCOPE_API_KEY")
    )
    return RerankerConfig(
        provider=settings.RERANK_PROVIDER,
        model=settings.RERANK_MODEL,
        api_key=api_key,
        base_url=settings.RERANK_BASE_URL,
        top_k=settings.RERANK_TOP_K,
        timeout=settings.RERANK_TIMEOUT,
        fail_open=settings.RERANK_FAIL_OPEN,
    )
