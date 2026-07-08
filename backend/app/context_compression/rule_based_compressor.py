import re
from typing import Any

from backend.app.context_compression.base import (
    BaseContextCompressor,
    CompressionInput,
    CompressionResult,
)


class RuleBasedContextCompressor(BaseContextCompressor):
    def __init__(self, max_chunk_chars: int = 1200) -> None:
        self.max_chunk_chars = max_chunk_chars

    def compress(self, input: CompressionInput) -> CompressionResult:
        original_chunks = input.chunks
        original_context_text = self._build_context_text(original_chunks)
        keywords = self._extract_keywords(input.query)
        sorted_chunks = sorted(
            original_chunks,
            key=self._chunk_rank_score,
            reverse=True,
        )

        compressed_chunks: list[Any] = []
        context_parts: list[str] = []
        current_chars = 0

        for chunk in sorted_chunks:
            compressed_text = self._compress_chunk_text(
                text=self._chunk_text(chunk),
                keywords=keywords,
            )
            if not compressed_text:
                continue

            chunk_metadata = self._build_chunk_metadata(chunk)
            compressed_chunk = self._copy_chunk(
                chunk=chunk,
                text=compressed_text,
                metadata=chunk_metadata,
            )
            context_part = self._format_context_chunk(compressed_chunk)
            separator_chars = 2 if context_parts else 0
            next_chars = current_chars + separator_chars + len(context_part)
            if next_chars > input.max_chars:
                continue

            compressed_chunks.append(compressed_chunk)
            context_parts.append(context_part)
            current_chars = next_chars

        compressed_context_text = "\n\n".join(context_parts)
        original_chunk_count = len(original_chunks)
        compressed_chunk_count = len(compressed_chunks)
        return CompressionResult(
            compressed_chunks=compressed_chunks,
            original_chunk_count=original_chunk_count,
            compressed_chunk_count=compressed_chunk_count,
            original_chars=len(original_context_text),
            compressed_chars=len(compressed_context_text),
            skipped_chunk_count=original_chunk_count - compressed_chunk_count,
            metadata={
                "provider": "rule_based",
                "max_chars": input.max_chars,
                "max_chunk_chars": self.max_chunk_chars,
                "keywords": keywords,
                "context_text": compressed_context_text,
            },
        )

    def _compress_chunk_text(self, text: str, keywords: list[str]) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        candidate_text = normalized_text
        matched_sentences = self._matched_sentences(normalized_text, keywords)
        if matched_sentences:
            candidate_text = "\n".join(matched_sentences)

        if len(candidate_text) <= self.max_chunk_chars:
            return candidate_text
        return candidate_text[: self.max_chunk_chars].rstrip()

    def _matched_sentences(self, text: str, keywords: list[str]) -> list[str]:
        if not keywords:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
            if sentence.strip()
        ]
        matched = [
            sentence
            for sentence in sentences
            if any(keyword.lower() in sentence.lower() for keyword in keywords)
        ]
        return matched

    def _extract_keywords(self, query: str) -> list[str]:
        keywords: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", query):
            normalized = token.lower()
            if normalized not in seen:
                seen.add(normalized)
                keywords.append(token)
        return keywords

    def _build_context_text(self, chunks: list[Any]) -> str:
        return "\n\n".join(self._format_context_chunk(chunk) for chunk in chunks)

    def _format_context_chunk(self, chunk: Any) -> str:
        return (
            f"[来源: {self._chunk_source(chunk)}, 文档ID: {self._chunk_document_id(chunk)}, "
            f"Chunk: {self._chunk_index(chunk)}]\n{self._chunk_text(chunk)}"
        )

    def _copy_chunk(self, chunk: Any, text: str, metadata: dict) -> Any:
        if hasattr(chunk, "model_copy"):
            return chunk.model_copy(update={"text": text, "metadata": metadata})
        if isinstance(chunk, dict):
            copied = dict(chunk)
            copied["text"] = text
            copied["metadata"] = metadata
            return copied
        return chunk

    def _build_chunk_metadata(self, chunk: Any) -> dict:
        metadata = {**self._chunk_metadata(chunk)}
        self._set_metadata_if_present(metadata, "source", self._chunk_source(chunk))
        self._set_metadata_if_present(metadata, "document_id", self._chunk_document_id(chunk))
        self._set_metadata_if_present(
            metadata,
            "knowledge_base_id",
            self._chunk_knowledge_base_id(chunk),
        )
        self._set_metadata_if_present(metadata, "chunk_index", self._chunk_index(chunk))
        self._set_metadata_if_present(metadata, "score", self._chunk_score(chunk))
        self._set_metadata_if_present(
            metadata,
            "rerank_score",
            self._chunk_rerank_score(chunk),
        )
        self._set_metadata_if_present(
            metadata,
            "rerank_rank",
            self._chunk_metadata_value(chunk, "rerank_rank"),
        )
        self._set_metadata_if_present(
            metadata,
            "rerank_provider",
            self._chunk_metadata_value(chunk, "rerank_provider"),
        )
        self._set_metadata_if_present(
            metadata,
            "rerank_model",
            self._chunk_metadata_value(chunk, "rerank_model"),
        )
        metadata["context_compressed"] = True
        metadata["context_compression_provider"] = "rule_based"
        return metadata

    def _chunk_rank_score(self, chunk: Any) -> float:
        score = self._chunk_rerank_score(chunk)
        if score is None:
            score = self._chunk_score(chunk)
        return score if score is not None else 0.0

    def _chunk_text(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text") or "")
        return str(getattr(chunk, "text", "") or "")

    def _chunk_metadata(self, chunk: Any) -> dict:
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata") or {}
        else:
            metadata = getattr(chunk, "metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    def _chunk_metadata_value(self, chunk: Any, key: str) -> Any | None:
        metadata = self._chunk_metadata(chunk)
        if key in metadata:
            return metadata[key]
        if isinstance(chunk, dict):
            return chunk.get(key)
        return getattr(chunk, key, None)

    def _set_metadata_if_present(self, metadata: dict, key: str, value: Any | None) -> None:
        if value is not None:
            metadata[key] = value

    def _chunk_source(self, chunk: Any) -> str | None:
        if isinstance(chunk, dict):
            source = chunk.get("source")
        else:
            source = getattr(chunk, "source", None)
        return source or self._chunk_metadata(chunk).get("source")

    def _chunk_document_id(self, chunk: Any) -> int | None:
        if isinstance(chunk, dict):
            value = chunk.get("document_id")
        else:
            value = getattr(chunk, "document_id", None)
        return value if isinstance(value, int) else None

    def _chunk_knowledge_base_id(self, chunk: Any) -> int | None:
        if isinstance(chunk, dict):
            value = chunk.get("knowledge_base_id")
        else:
            value = getattr(chunk, "knowledge_base_id", None)
        return value if isinstance(value, int) else None

    def _chunk_index(self, chunk: Any) -> int | None:
        if isinstance(chunk, dict):
            value = chunk.get("chunk_index")
        else:
            value = getattr(chunk, "chunk_index", None)
        return value if isinstance(value, int) else None

    def _chunk_score(self, chunk: Any) -> float | None:
        if isinstance(chunk, dict):
            value = chunk.get("score")
        else:
            value = getattr(chunk, "score", None)
        if value is None:
            value = self._chunk_metadata(chunk).get("score")
        return value if isinstance(value, int | float) else None

    def _chunk_rerank_score(self, chunk: Any) -> float | None:
        if isinstance(chunk, dict):
            value = chunk.get("rerank_score")
        else:
            value = getattr(chunk, "rerank_score", None)
        if value is None:
            value = self._chunk_metadata(chunk).get("rerank_score")
        return value if isinstance(value, int | float) else None
