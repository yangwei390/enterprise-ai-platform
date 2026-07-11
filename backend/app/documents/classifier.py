import re

from pydantic import BaseModel, Field


class DocumentClassificationResult(BaseModel):
    document_type: str
    confidence: float
    matched_rules: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DocumentClassifier:
    legal_patterns = [
        re.compile(r"中华人民共和国.{0,20}(法|条例|办法|规定)"),
        re.compile(r"第[零〇一二两三四五六七八九十百千万\d]+章"),
        re.compile(r"第[零〇一二两三四五六七八九十百千万\d]+条"),
    ]
    markdown_heading_pattern = re.compile(r"^#{1,6}\s+\S+", re.MULTILINE)
    markdown_fence_pattern = re.compile(r"```")

    def classify(
        self,
        *,
        text: str,
        filename: str | None = None,
        mime_type: str | None = None,
        metadata: dict | None = None,
    ) -> DocumentClassificationResult:
        try:
            return self._classify(
                text=text,
                filename=filename,
                mime_type=mime_type,
                metadata=metadata or {},
            )
        except Exception as exc:
            return DocumentClassificationResult(
                document_type="plain_text",
                confidence=0.0,
                matched_rules=[],
                metadata={"classification_failed": True, "error": str(exc)},
            )

    def _classify(
        self,
        *,
        text: str,
        filename: str | None,
        mime_type: str | None,
        metadata: dict,
    ) -> DocumentClassificationResult:
        suffix = str(metadata.get("suffix") or "")
        if filename and "." in filename:
            suffix = filename.rsplit(".", 1)[-1].lower()
        matched_rules: list[str] = []
        if suffix in {"md", "markdown"} or "markdown" in str(mime_type or "").lower():
            matched_rules.append("extension_or_mime_markdown")
        if self.markdown_heading_pattern.search(text):
            matched_rules.append("markdown_heading")
        if matched_rules:
            return DocumentClassificationResult(
                document_type="markdown",
                confidence=0.8,
                matched_rules=matched_rules,
                metadata={"suffix": suffix},
            )

        legal_matches = [
            pattern.pattern for pattern in self.legal_patterns if pattern.search(text)
        ]
        if legal_matches:
            score = min(0.95, 0.55 + 0.15 * len(legal_matches))
            return DocumentClassificationResult(
                document_type="legal",
                confidence=score,
                matched_rules=legal_matches,
                metadata={"suffix": suffix},
            )

        return DocumentClassificationResult(
            document_type="plain_text",
            confidence=0.5,
            matched_rules=["fallback_plain_text"],
            metadata={"suffix": suffix},
        )
