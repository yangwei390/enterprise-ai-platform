from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    text: str
    page_count: int | None = None
    language: str | None = None
    metadata: dict = Field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        raise NotImplementedError
