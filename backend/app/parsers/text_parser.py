import re
from pathlib import Path

from backend.app.parsers.base import BaseParser, DocumentElement, ParseQuality, ParseResult


class TextParser(BaseParser):

    def parse(self, file_path: Path) -> ParseResult:
        text = file_path.read_text(encoding="utf-8")
        elements = _parse_text_elements(text)
        value = ParseResult(
            text=text,
            metadata={
                "suffix": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
            },
            elements=elements,
            parse_quality=ParseQuality(
                page_count=None,
                element_count=len(elements),
            ),
        )
        return value


def _parse_text_elements(text: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    section_path: list[str] = []
    for line in text.splitlines():
        content = line.strip()
        if not content:
            continue
        element_type = _classify_line(content)
        if element_type == "heading":
            section_path = [content]
        elements.append(
            DocumentElement(
                type=element_type,
                content=content,
                section_path=list(section_path),
            )
        )
    return elements


def _classify_line(content: str) -> str:
    if re.match(r"^(#{1,6}\s+|第[一二三四五六七八九十百千万\d]+[章节]\b)", content):
        return "heading"
    if re.match(r"^([-*•]\s+|\d+[.)、]\s+|[一二三四五六七八九十]+[、.]\s*)", content):
        return "list"
    if len(content) <= 40 and not content.endswith(("。", "，", "；", ".", ",")):
        return "heading"
    return "paragraph"


if __name__ == '__main__':
    parser = TextParser()
    print(parser.parse(Path("/Users/yangwei/Desktop/Android.txt")))
