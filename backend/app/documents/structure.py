import re
from dataclasses import dataclass, field

CHINESE_NUMBER_PATTERN = r"[零〇一二两三四五六七八九十百千万\d]+"


def parse_chinese_number(value: str) -> int | None:
    raw = value.strip()
    raw = raw.removeprefix("第")
    for suffix in ("编", "章", "节", "条"):
        raw = raw.removesuffix(suffix)
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)

    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    section = 0
    number = 0
    for char in raw:
        if char in digits:
            number = digits[char]
            continue
        unit = units.get(char)
        if unit is None:
            return None
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = 0
        else:
            section += (number or 1) * unit
        number = 0
    return total + section + number


@dataclass
class StructureQueryHint:
    chapter_number: int | None = None
    chapter_label: str | None = None
    article_number: int | None = None
    article_label: str | None = None
    section_title: str | None = None
    heading_keywords: list[str] = field(default_factory=list)

    @property
    def has_hint(self) -> bool:
        return any(
            [
                self.chapter_number is not None,
                self.article_number is not None,
                self.section_title,
                self.heading_keywords,
            ]
        )

    def to_metadata(self) -> dict:
        return {
            "chapter_number": self.chapter_number,
            "chapter_label": self.chapter_label,
            "article_number": self.article_number,
            "article_label": self.article_label,
            "section_title": self.section_title,
            "heading_keywords": self.heading_keywords,
            "has_hint": self.has_hint,
        }


class StructureQueryHintParser:
    chapter_pattern = re.compile(rf"第(?P<number>{CHINESE_NUMBER_PATTERN})(?P<label>章|编|节)")
    article_pattern = re.compile(rf"第(?P<number>{CHINESE_NUMBER_PATTERN})条")

    def parse(self, query: str) -> StructureQueryHint:
        chapter_match = self.chapter_pattern.search(query)
        article_match = self.article_pattern.search(query)
        chapter_number = None
        chapter_label = None
        if chapter_match:
            chapter_number = parse_chinese_number(chapter_match.group("number"))
            chapter_label = chapter_match.group(0)

        article_number = None
        article_label = None
        if article_match:
            article_number = parse_chinese_number(article_match.group("number"))
            article_label = article_match.group(0)

        heading_keywords = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        return StructureQueryHint(
            chapter_number=chapter_number,
            chapter_label=chapter_label,
            article_number=article_number,
            article_label=article_label,
            heading_keywords=heading_keywords[:5],
        )
