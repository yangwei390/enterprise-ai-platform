from pathlib import Path

from backend.app.parsers.base import BaseParser, ParseResult


class TextParser(BaseParser):
    def parse(self, file_path: Path) -> ParseResult:
        text = file_path.read_text(encoding="utf-8")
        return ParseResult(
            text=text,
            metadata={
                "suffix": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
            },
        )
