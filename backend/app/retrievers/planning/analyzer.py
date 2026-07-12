import re
from time import perf_counter

from backend.app.retrievers.planning.constraint_registry import ConstraintRegistry
from backend.app.retrievers.planning.schemas import QueryAnalysisResult, RetrievalConstraint

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
    lexical_patterns = (
        re.compile(r"\b[A-Z]{1,8}-\d{2,}\b"),
        re.compile(r"\bHTTP\s*\d{3}\b", re.IGNORECASE),
        re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
        re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)"),
        re.compile(r"\b[A-Z][A-Za-z0-9_]+(?:Error|Exception|Service|Controller)\b"),
    )

    def __init__(self, registry: ConstraintRegistry) -> None:
        self.registry = registry

    def analyze(self, query: str, rewritten_query: str | None = None) -> QueryAnalysisResult:
        started = perf_counter()
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
