from pathlib import Path

from backend.app.db.session import SessionLocal
from backend.app.logger import logger
from backend.app.models import Document
from pydantic import BaseModel, Field
from sqlalchemy import select


class AutoMetadataFilterResult(BaseModel):
    candidate_document_ids: list[int] = Field(default_factory=list)
    source_hints: list[str] = Field(default_factory=list)
    auto_filter_applied: bool = False
    soft_boost_enabled: bool = False
    metadata: dict = Field(default_factory=dict)


class AutoMetadataFilterBuilder:
    min_hint_length = 3

    def build(
        self,
        query: str,
        knowledge_base_id: int | None,
        metadata_filter: dict | None = None,
    ) -> AutoMetadataFilterResult:
        if metadata_filter:
            return AutoMetadataFilterResult(
                metadata={
                    "skipped_reason": "explicit_metadata_filter_present",
                    "metadata_filter": metadata_filter,
                }
            )

        try:
            documents = self._load_documents(knowledge_base_id)
        except Exception as exc:
            logger.exception("Auto metadata filter document lookup failed")
            return AutoMetadataFilterResult(
                metadata={
                    "unavailable_reason": str(exc),
                }
            )

        normalized_query = self._normalize(query)
        candidate_document_ids: list[int] = []
        source_hints: list[str] = []
        matched_documents = []

        for document in documents:
            document_hints = self._build_document_hints(document)
            matched_hints = [
                hint for hint in document_hints if self._normalize(hint) in normalized_query
            ]
            if not matched_hints:
                continue

            identity = _document_identity(document)
            candidate_document_ids.append(document.id)
            source_hint = (
                identity.get("document_title")
                or document.original_filename
                or document.filename
                or document.storage_path
                or str(document.id)
            )
            source_hints.append(source_hint)
            matched_documents.append(
                {
                    "document_id": document.id,
                    "source": source_hint,
                    "matched_hints": matched_hints[:5],
                }
            )

        unique_candidate_ids = list(dict.fromkeys(candidate_document_ids))
        unique_source_hints = list(dict.fromkeys(source_hints))
        auto_filter_applied = bool(unique_candidate_ids)

        return AutoMetadataFilterResult(
            candidate_document_ids=unique_candidate_ids,
            source_hints=unique_source_hints,
            auto_filter_applied=auto_filter_applied,
            soft_boost_enabled=auto_filter_applied,
            metadata={
                "strategy": "document_identity_keyword_match",
                "matched_documents": matched_documents,
                "knowledge_base_id": knowledge_base_id,
            },
        )

    def _load_documents(self, knowledge_base_id: int | None) -> list[Document]:
        db = SessionLocal()
        try:
            statement = select(Document).where(Document.deleted_at.is_(None))
            if knowledge_base_id is not None:
                statement = statement.where(Document.knowledge_base_id == knowledge_base_id)
            result = db.execute(statement)
            return list(result.scalars().all())
        finally:
            db.close()

    def _build_document_hints(self, document: Document) -> list[str]:
        identity = _document_identity(document)
        raw_values = _identity_hint_values(identity)
        fallback_values = [
            document.original_filename,
            document.filename,
            Path(document.storage_path).name if document.storage_path else None,
            Path(document.storage_path).stem if document.storage_path else None,
        ]
        raw_values.extend(value for value in fallback_values if value)
        hints: list[str] = []
        for raw_value in raw_values:
            if not raw_value:
                continue
            value = str(raw_value)
            stem = Path(value).stem
            hints.extend([value, stem])
            hints.extend(self._chinese_substrings(stem))

        return [
            hint
            for hint in dict.fromkeys(hints)
            if len(self._normalize(hint)) >= self.min_hint_length
        ]

    def _chinese_substrings(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        chinese_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
        chinese_text = "".join(chinese_chars)
        substrings = []
        for start in range(len(chinese_text)):
            for end in range(start + self.min_hint_length, len(chinese_text) + 1):
                substrings.append(chinese_text[start:end])
        return substrings

    def _normalize(self, text: str) -> str:
        return (
            text.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("（", "")
            .replace("）", "")
            .replace("(", "")
            .replace(")", "")
        )


def _document_identity(document: Document) -> dict:
    document_metadata = document.document_metadata
    if isinstance(document_metadata, dict):
        nested = document_metadata.get("document_identity")
        return nested if isinstance(nested, dict) else {}
    return {}


def _identity_hint_values(identity: dict) -> list[str]:
    values: list[str] = []
    for key in ("document_title", "summary", "category", "language"):
        value = identity.get(key)
        if isinstance(value, str):
            values.append(value)
    for key in ("aliases", "keywords"):
        items = identity.get(key)
        if isinstance(items, list):
            values.extend(str(item) for item in items if item)
    entities = identity.get("entities")
    if isinstance(entities, dict):
        for entity_values in entities.values():
            if isinstance(entity_values, list):
                values.extend(str(item) for item in entity_values if item)
            elif isinstance(entity_values, str):
                values.append(entity_values)
    return values
