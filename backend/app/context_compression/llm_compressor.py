import time
from typing import Any

from backend.app.context_compression.base import (
    BaseContextCompressor,
    CompressionInput,
    CompressionResult,
)
from backend.app.context_compression.rule_based_compressor import (
    RuleBasedContextCompressor,
)
from backend.app.llms import BaseLLM, LLMFactory, LLMMessage, LLMRequest
from backend.app.logger import logger


class LLMContextCompressor(BaseContextCompressor):
    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0,
        max_chunk_chars: int = 1200,
        max_calls: int = 8,
        llm: BaseLLM | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_chunk_chars = max_chunk_chars
        self.max_calls = max_calls
        self.llm = llm

    def compress(self, input: CompressionInput) -> CompressionResult:
        started_at = time.perf_counter()
        original_chunks = input.chunks
        original_context_text = self._build_context_text(original_chunks)
        sorted_chunks = sorted(
            enumerate(original_chunks),
            key=lambda item: self._chunk_rank_score(item[1]),
            reverse=True,
        )

        compressed_chunks: list[Any] = []
        llm_calls = 0
        llm_empty_outputs = 0
        chunk_failures = 0
        llm_skipped_chunks = max(len(original_chunks) - self.max_calls, 0)

        for _original_index, chunk in sorted_chunks[: self.max_calls]:
            try:
                llm_calls += 1
                compressed_text = self._compress_chunk_with_llm(
                    query=input.query,
                    chunk_text=self._chunk_text(chunk),
                )
            except Exception as exc:
                chunk_failures += 1
                logger.exception("LLM context compression chunk failed")
                compressed_text = self._chunk_text(chunk)
                chunk_metadata = self._build_chunk_metadata(
                    chunk=chunk,
                    compressed_text=compressed_text,
                    chunk_failed=True,
                    error=str(exc),
                )
                compressed_chunk = self._copy_chunk(chunk, compressed_text, chunk_metadata)
                compressed_chunks.append(compressed_chunk)
                continue

            if not compressed_text:
                llm_empty_outputs += 1
                continue

            if len(compressed_text) > self.max_chunk_chars:
                compressed_text = compressed_text[: self.max_chunk_chars].rstrip()

            chunk_metadata = self._build_chunk_metadata(
                chunk=chunk,
                compressed_text=compressed_text,
                chunk_failed=False,
                error=None,
            )
            compressed_chunk = self._copy_chunk(chunk, compressed_text, chunk_metadata)
            compressed_chunks.append(compressed_chunk)

        fallback_used = False
        if not compressed_chunks:
            fallback_used = True
            compressed_chunks = original_chunks

        compressed_context_text = self._build_context_text(compressed_chunks)
        if len(compressed_context_text) > input.max_chars:
            fallback_used = True
            rule_result = RuleBasedContextCompressor(
                max_chunk_chars=self.max_chunk_chars
            ).compress(
                CompressionInput(
                    query=input.query,
                    chunks=compressed_chunks,
                    max_chars=input.max_chars,
                    metadata=input.metadata,
                )
            )
            compressed_chunks = rule_result.compressed_chunks
            compressed_context_text = str(rule_result.metadata.get("context_text") or "")
            if not compressed_chunks:
                compressed_chunks = original_chunks
                compressed_context_text = original_context_text

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        compressed_chunk_count = len(compressed_chunks)
        return CompressionResult(
            compressed_chunks=compressed_chunks,
            original_chunk_count=len(original_chunks),
            compressed_chunk_count=compressed_chunk_count,
            original_chars=len(original_context_text),
            compressed_chars=len(compressed_context_text),
            skipped_chunk_count=len(original_chunks) - compressed_chunk_count,
            metadata={
                "provider": "llm",
                "context_text": compressed_context_text,
                "llm_model": self.model,
                "llm_calls": llm_calls,
                "llm_skipped_chunks": llm_skipped_chunks,
                "llm_empty_outputs": llm_empty_outputs,
                "llm_duration_ms": duration_ms,
                "llm_chunk_failures": chunk_failures,
                "fallback_used": fallback_used,
                "max_chunk_chars": self.max_chunk_chars,
                "max_chars": input.max_chars,
            },
        )

    def _compress_chunk_with_llm(self, query: str, chunk_text: str) -> str:
        llm = self.llm or LLMFactory.get_llm()
        response = llm.chat(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "你是企业级 RAG 上下文压缩器。\n"
                            "请只基于给定 chunk，保留与用户问题相关的原文信息。\n"
                            "删除无关网页导航、页码、页眉页脚、重复内容。\n"
                            "不得添加原文没有的信息。\n"
                            "如果该 chunk 与问题无关，只返回空字符串。\n"
                            "输出必须是纯文本，不要解释。"
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=f"用户问题：\n{query}\n\nchunk text：\n{chunk_text}",
                    ),
                ],
                metadata={
                    "context_compression": True,
                    "provider": "llm",
                },
            )
        )
        return response.answer.strip()

    def _build_chunk_metadata(
        self,
        *,
        chunk: Any,
        compressed_text: str,
        chunk_failed: bool,
        error: str | None,
    ) -> dict:
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
        for key in ("rerank_rank", "rerank_provider", "rerank_model"):
            self._set_metadata_if_present(metadata, key, self._chunk_metadata_value(chunk, key))

        metadata["context_compressed"] = True
        metadata["context_compression_provider"] = "llm"
        metadata["context_compression_original_chars"] = len(self._chunk_text(chunk))
        metadata["context_compression_compressed_chars"] = len(compressed_text)
        if chunk_failed:
            metadata["context_compression_chunk_failed"] = True
            metadata["context_compression_chunk_error"] = error
        return metadata

    def _copy_chunk(self, chunk: Any, text: str, metadata: dict) -> Any:
        if hasattr(chunk, "model_copy"):
            return chunk.model_copy(update={"text": text, "metadata": metadata})
        if isinstance(chunk, dict):
            copied = dict(chunk)
            copied["text"] = text
            copied["metadata"] = metadata
            return copied
        return chunk

    def _build_context_text(self, chunks: list[Any]) -> str:
        return "\n\n".join(self._format_context_chunk(chunk) for chunk in chunks)

    def _format_context_chunk(self, chunk: Any) -> str:
        return (
            f"[来源: {self._chunk_source(chunk)}, 文档ID: {self._chunk_document_id(chunk)}, "
            f"Chunk: {self._chunk_index(chunk)}]\n{self._chunk_text(chunk)}"
        )

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
