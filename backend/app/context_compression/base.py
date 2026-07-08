from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class CompressionInput(BaseModel):
    query: str
    chunks: list[Any]
    max_chars: int
    metadata: dict = Field(default_factory=dict)


class CompressionResult(BaseModel):
    compressed_chunks: list[Any]
    original_chunk_count: int
    compressed_chunk_count: int
    original_chars: int
    compressed_chars: int
    skipped_chunk_count: int
    metadata: dict = Field(default_factory=dict)


class BaseContextCompressor(ABC):
    @abstractmethod
    def compress(self, input: CompressionInput) -> CompressionResult:
        raise NotImplementedError
