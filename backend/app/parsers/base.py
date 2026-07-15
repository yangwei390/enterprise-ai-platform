from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class TableStructure(BaseModel):
    title: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    units: str | None = None
    continuation_of: str | None = None


class DocumentElement(BaseModel):
    type: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class ParseQuality(BaseModel):
    page_count: int | None = None
    element_count: int = 0
    table_count: int = 0
    removed_repeated_header_footer_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    text: str
    page_count: int | None = None
    language: str | None = None
    metadata: dict = Field(default_factory=dict)
    elements: list[DocumentElement] = Field(default_factory=list)
    parse_quality: ParseQuality | None = None


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        raise NotImplementedError
