from typing import Any

from pydantic import BaseModel, Field


class ContextCompressionResult(BaseModel):
    context_text: str
    chunks: list[Any]
    original_chars: int
    compressed_chars: int
    compression_applied: bool
    metadata: dict = Field(default_factory=dict)
