from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.app.chunkers import ChunkResult
from backend.app.cleaners import CleanResult
from backend.app.parsers import ParseResult


@dataclass
class PipelineContext:
    document: Any
    parse_result: ParseResult | None = None
    clean_result: CleanResult | None = None
    chunk_result: ChunkResult | None = None
    metadata: dict = field(default_factory=dict)


class PipelineStep(ABC):
    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError
