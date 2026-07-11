from backend.app.chunkers.base import BaseChunker
from backend.app.chunkers.fixed_chunker import FixedChunker
from backend.app.chunkers.legal import LegalStructureChunker
from backend.app.chunkers.markdown import MarkdownChunker
from backend.app.chunkers.parent_child import ParentChildChunker
from backend.app.chunkers.recursive import RecursiveChunker
from backend.app.chunkers.semantic import SemanticChunker


class ChunkerFactory:
    @staticmethod
    def get_chunker(
        suffix: str | None = None,
        strategy: str | None = None,
    ) -> BaseChunker:
        selected = (strategy or "fixed").lower()
        if selected in {"legal", "legal_structure"}:
            return LegalStructureChunker()
        if selected == "markdown":
            return MarkdownChunker()
        if selected == "recursive":
            return RecursiveChunker()
        if selected == "semantic":
            return SemanticChunker()
        if selected == "parent_child":
            return ParentChildChunker()
        return FixedChunker()
