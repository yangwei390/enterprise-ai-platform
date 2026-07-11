from backend.app.documents.schemas import DocumentStructure
from backend.app.documents.structures import (
    BaseStructureParser,
    LegalStructureParser,
    MarkdownStructureParser,
    PlainTextStructureParser,
)


class StructureParserFactory:
    parsers: dict[str, type[BaseStructureParser]] = {
        "legal": LegalStructureParser,
        "markdown": MarkdownStructureParser,
        "plain_text": PlainTextStructureParser,
        "unknown": PlainTextStructureParser,
    }

    @classmethod
    def get_parser(cls, document_type: str) -> BaseStructureParser:
        parser_class = cls.parsers.get(document_type, PlainTextStructureParser)
        return parser_class()

    @classmethod
    def parse(cls, text: str, metadata: dict, document_type: str) -> DocumentStructure:
        return cls.get_parser(document_type).parse(text, metadata)
