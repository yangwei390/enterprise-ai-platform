import re

from backend.app.query.base import QueryRewriteResult


class SimpleQueryRewriter:
    prefixes = ("请问", "帮我查一下", "你知道")

    def rewrite(self, query: str) -> QueryRewriteResult:
        original_query = query
        rewritten_query = self._normalize_query(query)
        if not rewritten_query:
            rewritten_query = original_query

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            changed=rewritten_query != original_query,
            metadata={
                "rewriter": "simple",
                "prefixes": list(self.prefixes),
            },
        )

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip()
        normalized = normalized.replace("?", "？")
        normalized = re.sub(r"\s+", " ", normalized)

        for prefix in self.prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        return normalized
