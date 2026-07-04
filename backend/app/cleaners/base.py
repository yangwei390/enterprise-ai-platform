from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class CleanResult(BaseModel):
    text: str
    original_length: int
    cleaned_length: int
    metadata: dict = Field(default_factory=dict)


class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> CleanResult:
        raise NotImplementedError
