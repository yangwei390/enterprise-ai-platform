from backend.app.chunkers.base import BaseChunker
from backend.app.chunkers.fixed_chunker import FixedChunker


class ChunkerFactory:
    @staticmethod
    def get_chunker(suffix: str | None = None) -> BaseChunker:
        return FixedChunker()
