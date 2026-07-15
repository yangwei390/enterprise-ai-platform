import re
from time import perf_counter

from backend.app.retrievers.planning.constraint_registry import ConstraintRegistry
from backend.app.retrievers.planning.schemas import (
    QueryAnalysisResult,
    RetrievalConstraint,
    RetrievalOperator,
)
from backend.app.retrievers.query_understanding import QueryUnderstandingResult

_CHINESE_DIGITS = {
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
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


class FastQueryAnalyzer:
    chapter_pattern = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)章")
    article_pattern = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)条")
    section_pattern = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)节")
    heading_keyword_pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fff]{2,})(?:章节|章|节|标题|部分|制度|说明|流程)"
    )
    table_keyword_pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fff]{2,})(?:表|表格)"
    )
    lexical_patterns = (
        re.compile(r"\b[A-Z]{1,8}-\d{2,}\b"),
        re.compile(r"\bHTTP\s*\d{3}\b", re.IGNORECASE),
        re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
        re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)"),
        re.compile(r"\b[A-Z][A-Za-z0-9_]+(?:Error|Exception|Service|Controller)\b"),
    )

    def __init__(self, registry: ConstraintRegistry) -> None:
        self.registry = registry

    def analyze(
        self,
        query: str,
        rewritten_query: str | None = None,
        understanding: QueryUnderstandingResult | None = None,
    ) -> QueryAnalysisResult:
        started = perf_counter()
        if understanding is not None:
            return self._from_understanding(understanding, started)
        active_query = rewritten_query or query
        constraints: list[RetrievalConstraint] = []
        matched_rules: list[str] = []

        constraints.extend(
            self._extract_number_constraint(
                active_query,
                pattern=self.chapter_pattern,
                field="chapter_number",
                matched_rules=matched_rules,
            )
        )
        constraints.extend(
            self._extract_number_constraint(
                active_query,
                pattern=self.article_pattern,
                field="article_number",
                matched_rules=matched_rules,
            )
        )
        constraints.extend(
            self._extract_number_constraint(
                active_query,
                pattern=self.section_pattern,
                field="section_number",
                matched_rules=matched_rules,
            )
        )
        constraints.extend(self._extract_heading_constraints(active_query, matched_rules))
        constraints.extend(self._extract_table_constraints(active_query, matched_rules))

        lexical_score = (
            1.0
            if any(pattern.search(active_query) for pattern in self.lexical_patterns)
            else 0.0
        )
        if lexical_score:
            matched_rules.append("lexical_exact_token")

        if constraints:
            intent = "structured"
        elif lexical_score:
            intent = "lexical"
        else:
            intent = "hybrid"

        return QueryAnalysisResult(
            intent=intent,
            constraints=constraints,
            lexical_score=lexical_score,
            semantic_score=0.7 if intent == "hybrid" else 0.2,
            metadata={
                "analyzer": "fast_rule_based",
                "matched_rules": matched_rules,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "llm_called": False,
            },
        )

    def _from_understanding(
        self,
        understanding: QueryUnderstandingResult,
        started: float,
    ) -> QueryAnalysisResult:
        constraints: list[RetrievalConstraint] = []
        matched_rules: list[str] = []
        for hint in understanding.structure_hints:
            hint_type = hint.get("type")
            if hint_type == "chapter":
                constraints.extend(
                    self._constraint_from_hint(
                        field="chapter_number",
                        operator="eq",
                        value=hint.get("number"),
                        source_detail="query_understanding.chapter",
                        confidence=understanding.confidence,
                    )
                )
                matched_rules.append("chapter_number")
            elif hint_type == "article":
                constraints.extend(
                    self._constraint_from_hint(
                        field="article_number",
                        operator="eq",
                        value=hint.get("number"),
                        source_detail="query_understanding.article",
                        confidence=understanding.confidence,
                    )
                )
                matched_rules.append("article_number")
            elif hint_type == "section":
                constraints.extend(
                    self._constraint_from_hint(
                        field="section_number",
                        operator="eq",
                        value=hint.get("number"),
                        source_detail="query_understanding.section",
                        confidence=understanding.confidence,
                    )
                )
                matched_rules.append("section_number")
            elif hint_type == "heading":
                constraints.extend(
                    self._constraint_from_hint(
                        field="heading_title",
                        operator="contains",
                        value=hint.get("value"),
                        source_detail="query_understanding.heading",
                        confidence=min(understanding.confidence, 0.85),
                    )
                )
                matched_rules.append("heading_title")

        exact_tokens = understanding.metadata.get("exact_tokens")
        lexical_score = (
            1.0
            if understanding.intent == "lexical"
            or (isinstance(exact_tokens, list) and bool(exact_tokens))
            else 0.0
        )
        intent = "structured" if constraints else understanding.intent
        if intent == "open_query":
            intent = "hybrid"

        return QueryAnalysisResult(
            intent=intent,
            constraints=constraints,
            lexical_score=lexical_score,
            semantic_score=0.7 if intent in ("hybrid", "factual", "summary") else 0.2,
            metadata={
                "analyzer": "query_understanding",
                "matched_rules": matched_rules,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "llm_called": False,
                "query_understanding": understanding.model_dump(),
            },
        )

    def _constraint_from_hint(
        self,
        *,
        field: str,
        operator: RetrievalOperator,
        value,
        source_detail: str,
        confidence: float,
    ) -> list[RetrievalConstraint]:
        definition = self.registry.get(field)
        if definition is None or not definition.enabled or value in (None, ""):
            return []
        return [
            RetrievalConstraint(
                field=definition.field,
                operator=operator,
                value=value,
                confidence=confidence,
                source="query_understanding",
                source_detail=source_detail,
            )
        ]

    def _extract_number_constraint(
        self,
        query: str,
        *,
        pattern: re.Pattern[str],
        field: str,
        matched_rules: list[str],
    ) -> list[RetrievalConstraint]:
        definition = self.registry.get(field)
        if definition is None or not definition.enabled:
            return []
        constraints: list[RetrievalConstraint] = []
        for match in pattern.finditer(query):
            value = parse_chinese_or_arabic_number(match.group(1))
            if value is None:
                continue
            matched_rules.append(field)
            constraints.append(
                RetrievalConstraint(
                    field=definition.field,
                    operator="eq",
                    value=value,
                    confidence=1.0,
                    source="rule",
                    source_detail=field,
                )
            )
        return constraints

    def _extract_heading_constraints(
        self,
        query: str,
        matched_rules: list[str],
    ) -> list[RetrievalConstraint]:
        definition = self.registry.get("heading_title")
        if definition is None or not definition.enabled:
            return []
        constraints: list[RetrievalConstraint] = []
        for match in self.heading_keyword_pattern.finditer(query):
            if re.fullmatch(
                r"第[0-9零〇一二两三四五六七八九十百千万]+[章节]",
                match.group(0),
            ):
                continue
            for keyword in _heading_keywords(match.group(0), match.group(1)):
                matched_rules.append("heading_title")
                constraints.append(
                    RetrievalConstraint(
                        field=definition.field,
                        operator="contains",
                        value=keyword,
                        confidence=0.75,
                        source="rule",
                        source_detail="heading_keyword",
                    )
                )
        return constraints

    def _extract_table_constraints(
        self,
        query: str,
        matched_rules: list[str],
    ) -> list[RetrievalConstraint]:
        definition = self.registry.get("table_title")
        if definition is None or not definition.enabled:
            return []
        constraints: list[RetrievalConstraint] = []
        for match in self.table_keyword_pattern.finditer(query):
            keyword = _normalize_keyword(match.group(1))
            if not keyword:
                continue
            matched_rules.append("table_title")
            constraints.append(
                RetrievalConstraint(
                    field=definition.field,
                    operator="contains",
                    value=keyword,
                    confidence=0.75,
                    source="rule",
                    source_detail="table_keyword",
                )
            )
        return constraints


def parse_chinese_or_arabic_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS.get(char)
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


def _normalize_keyword(value: str) -> str:
    normalized = value.strip(" 的关于请问一下中")
    if len(normalized) > 12:
        normalized = normalized[-12:]
    return normalized


def _heading_keywords(full_match: str, captured: str) -> list[str]:
    candidates = []
    normalized = _normalize_keyword(captured)
    if normalized:
        candidates.append(normalized)
    if full_match.endswith("制度"):
        candidates.append(_normalize_keyword(full_match[-6:]))
        candidates.append(_normalize_keyword(full_match[-4:]))
    if full_match.endswith(("章节", "标题", "部分", "说明", "流程")):
        candidates.append(normalized)
    deduped = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
