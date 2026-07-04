from pathlib import Path

import fitz

from backend.app.parsers.base import BaseParser, ParseResult


class PdfParser(BaseParser):
    def parse(self, file_path: Path) -> ParseResult:
        texts: list[str] = []
        with fitz.open(file_path) as document:
            page_count = document.page_count
            for page in document:
                texts.append(page.get_text())

        return ParseResult(
            text="\n".join(texts),
            page_count=page_count,
            metadata={
                "suffix": file_path.suffix.lower(),
                "page_count": page_count,
            },
        )
