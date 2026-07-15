from __future__ import annotations

import re
from time import perf_counter
from typing import TYPE_CHECKING

from backend.app.query.base import QueryRewriteResult, RewriteType

if TYPE_CHECKING:
    from backend.app.retrievers.query_understanding import QueryUnderstandingResult


class SimpleQueryRewriter:
    oral_prefixes = (
        "请问",
        "帮我看看",
        "帮我查一下",
        "帮我查查",
        "我想知道",
        "能告诉我",
        "你知道",
        "麻烦看下",
    )
    summary_expansion_terms = ("主要内容", "概述")
    comparison_terms = ("区别", "相同点", "不同点")
    protected_file_pattern = re.compile(r"\b[\w.\-]+\.(?:pdf|docx?|xlsx?|pptx?|txt|md)\b", re.I)
    protected_version_pattern = re.compile(
        r"\b(?:v?\d+\.\d+(?:\.\d+)?|qwen-[\w.-]+|gpt-[\w.-]+)\b",
        re.I,
    )
    protected_code_pattern = re.compile(r"\b(?:ERR|ERROR|HTTP|ORA)[-_ ]?\d{2,}\b", re.I)
    protected_numbering_pattern = re.compile(
        r"(?:第[0-9零〇一二两三四五六七八九十百千万]+[章节条])|\b\d+\.\d+(?:\.\d+)*\b"
    )

    def rewrite(
        self,
        query: str,
        understanding: QueryUnderstandingResult | None = None,
        max_length: int | None = None,
    ) -> QueryRewriteResult:
        started = perf_counter()
        original_query = query
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            normalized_query = original_query

        if understanding is not None and self._must_preserve_query(
            original_query, understanding
        ):
            return self._result(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_type="NONE",
                rewrite_reason="protected_query",
                started=started,
                understanding=understanding,
            )

        rewritten_query = normalized_query
        rewrite_type: RewriteType = "NORMALIZATION" if rewritten_query != original_query else "NONE"
        rewrite_reason = "normalized_query" if rewrite_type != "NONE" else "no_rewrite_needed"

        if understanding is not None:
            enriched_query, enriched_type, enriched_reason = self._enrich_with_understanding(
                rewritten_query,
                understanding,
            )
            if enriched_query != rewritten_query:
                rewritten_query = enriched_query
                rewrite_type = enriched_type
                rewrite_reason = enriched_reason

        if max_length is not None and max_length > 0 and len(rewritten_query) > max_length:
            rewritten_query = rewritten_query[:max_length].rstrip()
            rewrite_type = "NORMALIZATION"
            rewrite_reason = "max_length_truncated"

        return self._result(
            original_query=original_query,
            rewritten_query=rewritten_query or original_query,
            rewrite_type=rewrite_type,
            rewrite_reason=rewrite_reason,
            started=started,
            understanding=understanding,
        )

    def _result(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        rewrite_type: RewriteType,
        rewrite_reason: str,
        started: float,
        understanding: QueryUnderstandingResult | None,
    ) -> QueryRewriteResult:
        changed = rewritten_query != original_query

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_reason=rewrite_reason,
            rewrite_type=rewrite_type if changed else "NONE",
            rewrite_changed=changed,
            changed=rewritten_query != original_query,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            metadata={
                "rewriter": "simple",
                "understanding_used": understanding is not None,
                "intent": understanding.intent if understanding is not None else None,
                "document_hints": (
                    understanding.document_hints if understanding is not None else []
                ),
                "structure_hints": (
                    understanding.structure_hints if understanding is not None else []
                ),
            },
        )

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip()
        normalized = normalized.replace("?", "？")
        normalized = re.sub(r"\s+", " ", normalized)

        for prefix in self.oral_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        normalized = self._remove_trailing_fillers(normalized)
        normalized = self._space_structure_tokens(normalized)
        return normalized

    def _must_preserve_query(
        self,
        query: str,
        understanding: QueryUnderstandingResult,
    ) -> bool:
        if understanding.intent == "lexical":
            return True
        return any(
            pattern.search(query)
            for pattern in (
                self.protected_file_pattern,
                self.protected_version_pattern,
                self.protected_code_pattern,
            )
        )

    def _enrich_with_understanding(
        self,
        query: str,
        understanding: QueryUnderstandingResult,
    ) -> tuple[str, RewriteType, str]:
        if understanding.intent == "structured":
            return query, "NORMALIZATION", "structured_query_normalization_only"
        if understanding.intent == "lexical":
            return query, "NONE", "lexical_query_not_rewritten"
        if understanding.intent == "summary":
            return self._append_missing_term(
                query,
                self.summary_expansion_terms,
                "主要内容",
                "EXPANSION",
                "summary_intent_expansion",
            )
        if understanding.intent in ("comparison", "multi_document"):
            return self._append_missing_term(
                query,
                self.comparison_terms,
                "区别",
                "KEYWORD_ENRICHMENT",
                "comparison_keyword_enrichment",
            )
        return query, "NONE", "no_rewrite_needed"

    def _append_missing_term(
        self,
        query: str,
        existing_terms: tuple[str, ...],
        term: str,
        rewrite_type: RewriteType,
        reason: str,
    ) -> tuple[str, RewriteType, str]:
        if any(existing in query for existing in existing_terms):
            return query, "NONE", "keyword_already_present"
        return f"{query} {term}", rewrite_type, reason

    def _remove_trailing_fillers(self, query: str) -> str:
        normalized = query
        for suffix in (
            "讲什么？",
            "讲什么",
            "讲的是什么？",
            "讲的是什么",
            "是什么？",
            "是什么",
            "吗",
            "呢",
            "？",
        ):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break
        return normalized

    def _space_structure_tokens(self, query: str) -> str:
        normalized = query
        normalized = self.protected_numbering_pattern.sub(
            lambda match: f" {match.group(0)} ",
            normalized,
        )
        return re.sub(r"\s+", " ", normalized).strip()
