from functools import lru_cache

from backend.app.config.settings import settings
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    provider: str = "dummy"
    model: str = "text-embedding-v4"
    api_key: str | None = None
    base_url: str | None = None
    dimension: int | None = None
    batch_size: int = 10
    metadata: dict = Field(default_factory=dict)


@lru_cache
def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=settings.EMBEDDING_PROVIDER,
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
        dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        metadata={
            "source": "settings",
        },
    )
