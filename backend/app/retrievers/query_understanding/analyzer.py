from time import perf_counter

from backend.app.retrievers.query_understanding.models import (
    QueryIntent,
    QueryUnderstandingResult,
)
from backend.app.retrievers.query_understanding.rules import (
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    DOCUMENT_CONTEXT_HINT_PATTERN,
    DOCUMENT_FILE_PATTERN,
    ENTITY_PATTERNS,
    EXACT_TOKEN_PATTERNS,
    HEADING_HINT_PATTERN,
    NEGATIVE_PATTERN,
    QUOTED_DOCUMENT_HINT_PATTERN,
    SECTION_PATTERN,
    STOP_WORDS,
    YEAR_PATTERN,
    YEAR_RANGE_PATTERN,
    normalize_query,
    parse_chinese_or_arabic_number,
    unique_non_empty,
)


class FastQueryAnalyzer:
    def analyze(self, query: str) -> QueryUnderstandingResult:
        started = perf_counter()
        normalized_query = normalize_query(query)
        structure_hints = self._extract_structure_hints(query)
        document_hints = self._extract_document_hints(query)
        temporal_constraints = self._extract_temporal_constraints(query)
        comparison_targets = self._extract_comparison_targets(query)
        negative_constraints = self._extract_negative_constraints(query)
        entities = self._extract_entities(query)
        exact_tokens = self._extract_exact_tokens(query)
        keywords = self._extract_keywords(query, document_hints, structure_hints, exact_tokens)
        intent = self._classify_intent(
            query=query,
            structure_hints=structure_hints,
            comparison_targets=comparison_targets,
            document_hints=document_hints,
            exact_tokens=exact_tokens,
        )
        confidence = self._confidence(
            intent=intent,
            structure_hints=structure_hints,
            document_hints=document_hints,
            keywords=keywords,
            exact_tokens=exact_tokens,
            temporal_constraints=temporal_constraints,
            comparison_targets=comparison_targets,
            negative_constraints=negative_constraints,
        )

        return QueryUnderstandingResult(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            keywords=keywords,
            entities=entities,
            document_hints=document_hints,
            structure_hints=structure_hints,
            temporal_constraints=temporal_constraints,
            comparison_targets=comparison_targets,
            negative_constraints=negative_constraints,
            confidence=confidence,
            analyzer_source="fast_rule_based",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            metadata={
                "llm_called": False,
                "exact_tokens": exact_tokens,
            },
        )

    def _extract_structure_hints(self, query: str) -> list[dict]:
        hints: list[dict] = []
        for pattern, hint_type in (
            (CHAPTER_PATTERN, "chapter"),
            (ARTICLE_PATTERN, "article"),
            (SECTION_PATTERN, "section"),
        ):
            for match in pattern.finditer(query):
                number = parse_chinese_or_arabic_number(match.group(1))
                if number is None:
                    continue
                hints.append(
                    {
                        "type": hint_type,
                        "number": number,
                        "raw": match.group(0),
                        "source": "rule",
                    }
                )
        for match in HEADING_HINT_PATTERN.finditer(query):
            value = match.group(1).strip("的关于")
            if len(value) < 2:
                continue
            hints.append(
                {
                    "type": "heading",
                    "value": value,
                    "raw": match.group(0),
                    "source": "rule",
                }
            )
        return hints

    def _extract_document_hints(self, query: str) -> list[str]:
        normalized = query
        for separator in ("和", "与", "及", "、", "，", ",", "对比", "比较"):
            normalized = normalized.replace(separator, "|")
        hints: list[str] = []
        hints.extend(
            _clean_document_hint(match.group(1))
            for match in DOCUMENT_FILE_PATTERN.finditer(query)
        )
        hints.extend(
            _clean_document_hint(match.group(1))
            for match in QUOTED_DOCUMENT_HINT_PATTERN.finditer(query)
        )
        for part in normalized.split("|"):
            part = _strip_structure_tokens(part)
            if not part:
                continue
            match = DOCUMENT_CONTEXT_HINT_PATTERN.match(part)
            if match is None:
                continue
            hint = _clean_document_hint(match.group(1))
            if _looks_like_document_hint(hint):
                hints.append(hint)
        return unique_non_empty(hints)

    def _extract_temporal_constraints(self, query: str) -> list[dict]:
        constraints: list[dict] = []
        for match in YEAR_RANGE_PATTERN.finditer(query):
            constraints.append(
                {
                    "type": "year_range",
                    "start": int(match.group(1)),
                    "end": int(match.group(2)),
                    "raw": match.group(0),
                }
            )
        for match in YEAR_PATTERN.finditer(query):
            year = int(match.group(1))
            if any(_year_in_range(year, item) for item in constraints):
                continue
            constraints.append({"type": "year", "value": year, "raw": match.group(0)})
        if "最近" in query or "近" in query and "年" in query:
            constraints.append({"type": "relative", "value": "recent", "raw": "最近"})
        return constraints

    def _extract_comparison_targets(self, query: str) -> list[str]:
        if not any(token in query for token in ("对比", "区别", "相同点", "不同点", "比较")):
            return []
        separators = ("和", "与", "及", "、", " vs ", " VS ", "相比")
        normalized = query
        for separator in separators:
            normalized = normalized.replace(separator, "|")
        parts = [_clean_comparison_part(part) for part in normalized.split("|")]
        return unique_non_empty([part for part in parts if len(part) >= 2])[:4]

    def _extract_negative_constraints(self, query: str) -> list[str]:
        return unique_non_empty([match.group(1) for match in NEGATIVE_PATTERN.finditer(query)])

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        entities: dict[str, list[str]] = {}
        for entity_type, pattern in ENTITY_PATTERNS.items():
            values = unique_non_empty([match.group(0) for match in pattern.finditer(query)])
            if values:
                entities[entity_type] = values
        return entities

    def _extract_exact_tokens(self, query: str) -> list[str]:
        tokens: list[str] = []
        for pattern in EXACT_TOKEN_PATTERNS:
            tokens.extend(match.group(0) for match in pattern.finditer(query))
        return unique_non_empty(tokens)

    def _extract_keywords(
        self,
        query: str,
        document_hints: list[str],
        structure_hints: list[dict],
        exact_tokens: list[str],
    ) -> list[str]:
        values = [*document_hints, *exact_tokens]
        values.extend(
            str(hint["value"])
            for hint in structure_hints
            if hint.get("type") == "heading" and hint.get("value")
        )
        chinese_parts = [
            part
            for part in query.replace("？", " ").replace("?", " ").split()
            if 2 <= len(part) <= 20 and part not in STOP_WORDS
        ]
        if not chinese_parts:
            chinese_parts = [
                query.strip(" ，。？！")
            ] if 2 <= len(query.strip(" ，。？！")) <= 20 else []
        values.extend(chinese_parts)
        return unique_non_empty(values)[:15]

    def _classify_intent(
        self,
        *,
        query: str,
        structure_hints: list[dict],
        comparison_targets: list[str],
        document_hints: list[str],
        exact_tokens: list[str],
    ) -> QueryIntent:
        if comparison_targets:
            return "comparison" if len(document_hints) < 2 else "multi_document"
        if structure_hints:
            return "structured"
        if any(token in query for token in ("总结", "概括", "主要内容", "讲的是什么")):
            return "summary"
        if exact_tokens:
            return "lexical"
        if len(document_hints) >= 2:
            return "multi_document"
        if any(token in query for token in ("是什么", "多少", "谁", "何时", "哪里", "为什么")):
            return "factual"
        return "open_query"

    def _confidence(
        self,
        *,
        intent: QueryIntent,
        structure_hints: list[dict],
        document_hints: list[str],
        keywords: list[str],
        exact_tokens: list[str],
        temporal_constraints: list[dict],
        comparison_targets: list[str],
        negative_constraints: list[str],
    ) -> float:
        score = 0.35
        if intent != "open_query":
            score += 0.15
        if structure_hints:
            score += 0.2
        if exact_tokens:
            score += 0.2
        if document_hints:
            score += 0.1
        if keywords:
            score += 0.05
        if temporal_constraints or comparison_targets or negative_constraints:
            score += 0.1
        return round(min(score, 0.95), 2)


def _year_in_range(year: int, item: dict) -> bool:
    start = item.get("start")
    end = item.get("end")
    return isinstance(start, int) and isinstance(end, int) and start <= year <= end


def _clean_document_hint(value: str) -> str:
    cleaned = value.strip(" ，。？！、:：；;（）()[]【】\"'")
    for prefix in (
        "概括",
        "总结",
        "查看",
        "查询",
        "请问",
        "帮我看看",
        "帮我查一下",
        "我想知道",
        "能告诉我",
        "关于",
        "对比",
        "比较",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    for suffix in ("主要内容", "不同点", "相同点", "区别"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    if cleaned.endswith("的"):
        cleaned = cleaned[:-1]
    return cleaned


def _strip_structure_tokens(value: str) -> str:
    cleaned = value.strip(" ，。？！、:：；;（）()[]【】\"'")
    for pattern in (CHAPTER_PATTERN, ARTICLE_PATTERN, SECTION_PATTERN):
        cleaned = pattern.sub("", cleaned)
    for token in ("讲什么", "讲的是什么", "是什么", "有哪些", "怎么规定", "怎么做"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip(" ，。？！、:：；;（）()[]【】\"'")


def _looks_like_document_hint(value: str) -> bool:
    if not (2 <= len(value) <= 80):
        return False
    if value in STOP_WORDS:
        return False
    if any(char.isspace() for char in value) and "." not in value:
        return len(value.split()) <= 6
    return True


def _clean_comparison_part(value: str) -> str:
    cleaned = value.strip(" ，。？！、:：；;（）()[]【】\"'")
    for token in ("什么", "有哪些", "的区别", "区别", "对比", "相同点", "不同点", "比较"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip(" ，。？！、:：；;（）()[]【】\"'")
