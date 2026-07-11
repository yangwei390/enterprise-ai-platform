from backend.app.documents.structures.base import BaseStructureParser
from backend.app.documents.structures.legal import LegalStructureParser
from backend.app.documents.structures.markdown import MarkdownStructureParser
from backend.app.documents.structures.plain_text import PlainTextStructureParser

__all__ = [
    "BaseStructureParser",
    "LegalStructureParser",
    "MarkdownStructureParser",
    "PlainTextStructureParser",
]
