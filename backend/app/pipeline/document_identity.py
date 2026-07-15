import re
from collections import Counter
from time import perf_counter

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.parsers import ParseResult
from pydantic import BaseModel, Field


class DocumentIdentity(BaseModel):
    document_title: str
    aliases: list[str] = Field(default_factory=list)
    summary: str
    keywords: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    language: str | None = None
    category: str = "general"
    identity_source: str = "rule"
    llm_used: bool = False
    duration_ms: float = 0.0


class DocumentIdentityAnalyzer:
    def analyze(
        self,
        *,
        parse_result: ParseResult,
        cleaned_text: str,
        filename: str | None = None,
    ) -> DocumentIdentity:
        started = perf_counter()
        elements = parse_result.elements or []
        headings = [
            element.content.strip()
            for element in elements
            if element.type == "heading" and element.content.strip()
        ]
        text = cleaned_text or parse_result.text
        language = parse_result.language or _detect_language(text)
        title, source = self._title(parse_result, headings, filename)
        keywords = self._keywords(title=title, headings=headings, text=text)
        entities = self._entities(text=text, title=title)
        aliases = self._aliases(title=title, language=language)
        category = self._category(title=title, keywords=keywords, entities=entities)
        summary = self._summary(title=title, headings=headings, text=text)
        identity = DocumentIdentity(
            document_title=title,
            aliases=aliases,
            summary=summary,
            keywords=keywords,
            entities=entities,
            language=language,
            category=category,
            identity_source=source,
            llm_used=False,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        logger.info(
            "Document identity | title=%s | aliases=%s | keywords=%s | entities=%s | "
            "category=%s | language=%s | source=%s | llm_used=%s | duration_ms=%s",
            identity.document_title,
            identity.aliases,
            identity.keywords,
            identity.entities,
            identity.category,
            identity.language,
            identity.identity_source,
            identity.llm_used,
            identity.duration_ms,
        )
        return identity

    def _title(
        self,
        parse_result: ParseResult,
        headings: list[str],
        filename: str | None,
    ) -> tuple[str, str]:
        metadata_title = str(parse_result.metadata.get("document_title") or "").strip()
        if headings:
            return headings[0], "element_heading"
        if metadata_title:
            return metadata_title, "parser_metadata"
        if getattr(settings, "DOCUMENT_IDENTITY_LLM_ENABLED", False):
            # Placeholder for later provider-backed extraction; fail closed to rule-based.
            pass
        fallback = _filename_stem(filename)
        return fallback or "Untitled Document", "filename_fallback"

    def _keywords(self, *, title: str, headings: list[str], text: str) -> list[str]:
        candidates: list[str] = []
        candidates.extend(_keyword_phrases(title))
        for heading in headings:
            candidates.extend(_keyword_phrases(heading))
        candidates.extend(_frequent_terms(text))
        return _dedupe(candidates)[:15]

    def _summary(self, *, title: str, headings: list[str], text: str) -> str:
        sentences = _sentences(text)
        pieces = []
        if title:
            pieces.append(f"本文档主题为{title}。")
        if headings:
            pieces.append("主要结构包括：" + "、".join(headings[:6]) + "。")
        pieces.extend(sentences[:3])
        summary = "".join(pieces).strip()
        if len(summary) > 300:
            summary = summary[:297].rstrip() + "..."
        return summary

    def _aliases(self, *, title: str, language: str | None) -> list[str]:
        aliases: list[str] = []
        normalized = title.strip()
        if normalized.startswith("中华人民共和国") and len(normalized) > 7:
            aliases.append(normalized.removeprefix("中华人民共和国"))
        if normalized.endswith("法"):
            aliases.append(f"{normalized}规")
        return _dedupe([alias for alias in aliases if alias and alias != title])[:8]

    def _entities(self, *, text: str, title: str) -> dict[str, list[str]]:
        combined = f"{title}\n{text}"
        laws = _dedupe(re.findall(r"(?:中华人民共和国)?[\u4e00-\u9fff]{2,20}法", combined))[:10]
        organizations = _dedupe(
            re.findall(r"[\u4e00-\u9fff]{2,30}(?:公司|部门|委员会|人民政府|法院|机构)", combined)
        )[:10]
        dates = _dedupe(
            re.findall(r"\d{4}年\d{1,2}月\d{1,2}日|\d{4}年|\d{1,2}月\d{1,2}日", combined)
        )[:10]
        locations = _dedupe(re.findall(r"[\u4e00-\u9fff]{2,8}(?:省|市|区|县)", combined))[:10]
        return {
            "laws": laws,
            "organizations": organizations,
            "people": [],
            "products": [],
            "departments": organizations,
            "locations": locations,
            "times": dates,
        }

    def _category(self, *, title: str, keywords: list[str], entities: dict[str, list[str]]) -> str:
        joined = " ".join([title, *keywords])
        if entities.get("laws") or "法律" in joined or joined.endswith("法"):
            return "legal"
        if "手册" in joined or "制度" in joined:
            return "policy"
        if "报告" in joined or "财务" in joined:
            return "report"
        return "general"


def _detect_language(text: str) -> str | None:
    if not text:
        return None
    cjk_count = sum(1 for char in text[:1000] if "\u4e00" <= char <= "\u9fff")
    ascii_count = sum(1 for char in text[:1000] if char.isascii() and char.isalpha())
    if cjk_count >= ascii_count:
        return "zh"
    if ascii_count:
        return "en"
    return None


def _filename_stem(filename: str | None) -> str | None:
    if not filename:
        return None
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.strip() or None


def _keyword_phrases(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text).strip()
    parts = [part for part in normalized.split() if len(part) >= 2]
    if not parts and len(normalized) >= 2:
        parts = [normalized]
    return parts


def _frequent_terms(text: str) -> list[str]:
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    english_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", text)
    counter = Counter(chinese_terms + english_terms)
    stop_words = {"本文", "文档", "以及", "进行", "相关", "包括", "一个"}
    return [
        term
        for term, _ in counter.most_common(30)
        if term not in stop_words and not term.isdigit()
    ]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [part for part in parts if 8 <= len(part) <= 160]


def _dedupe(items: list[str]) -> list[str]:
    result = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result
