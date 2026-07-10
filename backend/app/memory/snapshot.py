from pydantic import BaseModel, Field


class MemorySnapshot(BaseModel):
    provider: str
    session_count: int | None = None
    cache_count: int | None = None
    checkpoint_count: int | None = None
    metadata: dict = Field(default_factory=dict)
