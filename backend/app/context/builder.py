from __future__ import annotations

from time import perf_counter

from backend.app.context.base import (
    BaseContextBuilder,
    ContextBuildRequest,
    ContextBuildResult,
    ContextChunk,
)
from backend.app.context.formatter import format_context_chunk
from backend.app.rerankers import RerankedChunk


class BasicContextBuilder(BaseContextBuilder):
    overlap_threshold = 0.86

    def build(self, request: ContextBuildRequest) -> ContextBuildResult:
        started = perf_counter()
        try:
            return self._build(request, started)
        except Exception as exc:
            return self._build_fail_open(request, started, exc)

    def _build(
        self,
        request: ContextBuildRequest,
        started: float | None = None,
    ) -> ContextBuildResult:
        started = started or perf_counter()
        deduped, deduplicated_count, merged_count = self._deduplicate(request.chunks)
        organized = self._organize_chunks(deduped, request)
        selected, skipped_count, truncated = self._apply_budget(organized, request)
        context_text = "\n\n".join(format_context_chunk(chunk) for chunk in selected)
        context_chunks = [self._to_context_chunk(chunk) for chunk in selected]
        citations = [self._citation(chunk) for chunk in context_chunks]
        document_groups = self._document_groups(context_chunks)

        return ContextBuildResult(
            query=request.query,
            context_text=context_text,
            chunks=context_chunks,
            selected_chunks=context_chunks,
            citations=citations,
            document_groups=document_groups,
            total_chunks=len(context_chunks),
            total_chars=len(context_text),
            truncated=truncated,
            deduplicated_count=deduplicated_count,
            merged_count=merged_count,
            skipped_count=skipped_count,
            metadata=self._metadata(
                request=request,
                started=started,
                input_count=len(request.chunks),
                output_count=len(context_chunks),
                document_groups=document_groups,
                deduplicated_count=deduplicated_count,
                merged_count=merged_count,
                skipped_count=skipped_count,
                truncated=truncated,
                failed=False,
                error=None,
            ),
        )

    def _build_fail_open(
        self,
        request: ContextBuildRequest,
        started: float,
        exc: Exception,
    ) -> ContextBuildResult:
        selected, skipped_count, truncated = self._simple_select(request)
        context_text = "\n\n".join(format_context_chunk(chunk) for chunk in selected)
        context_chunks = [self._to_context_chunk(chunk) for chunk in selected]
        citations = [self._citation(chunk) for chunk in context_chunks]
        document_groups = self._document_groups(context_chunks)

        return ContextBuildResult(
            query=request.query,
            context_text=context_text,
            chunks=context_chunks,
            selected_chunks=context_chunks,
            citations=citations,
            document_groups=document_groups,
            total_chunks=len(context_chunks),
            total_chars=len(context_text),
            truncated=truncated,
            deduplicated_count=0,
            merged_count=0,
            skipped_count=skipped_count,
            metadata=self._metadata(
                request=request,
                started=started,
                input_count=len(request.chunks),
                output_count=len(context_chunks),
                document_groups=document_groups,
                deduplicated_count=0,
                merged_count=0,
                skipped_count=skipped_count,
                truncated=truncated,
                failed=True,
                error=str(exc),
            ),
        )

    def _deduplicate(
        self,
        chunks: list[RerankedChunk],
    ) -> tuple[list[RerankedChunk], int, int]:
        selected: list[RerankedChunk] = []
        keys: dict[tuple, int] = {}
        deduplicated_count = 0
        merged_count = 0

        for chunk in chunks:
            key = self._chunk_identity(chunk)
            existing_index = keys.get(key)
            if existing_index is not None:
                selected[existing_index] = self._choose_better(
                    selected[existing_index],
                    chunk,
                )
                deduplicated_count += 1
                continue

            overlap_index = self._overlap_index(selected, chunk)
            if overlap_index is not None:
                selected[overlap_index] = self._merge_chunks(
                    selected[overlap_index],
                    chunk,
                )
                merged_count += 1
                continue

            keys[key] = len(selected)
            selected.append(chunk)

        return selected, deduplicated_count, merged_count

    def _organize_chunks(
        self,
        chunks: list[RerankedChunk],
        request: ContextBuildRequest,
    ) -> list[RerankedChunk]:
        relevance_sorted = sorted(chunks, key=self._priority_key, reverse=True)
        if (
            request.strategy == "MULTI_DOCUMENT"
            and request.route_type == "MULTI_DOCUMENT"
            and request.multi_document_diversity_enabled
        ):
            return self._apply_multi_document_quota(
                relevance_sorted,
                target_document_ids=request.target_document_ids,
                max_per_document=request.multi_document_max_per_document,
                min_documents=request.multi_document_min_documents,
            )
        return self._restore_document_order(
            relevance_sorted,
            request.target_document_ids,
        )

    def _apply_budget(
        self,
        chunks: list[RerankedChunk],
        request: ContextBuildRequest,
    ) -> tuple[list[RerankedChunk], int, bool]:
        selected: list[RerankedChunk] = []
        skipped_count = 0
        total_chars = 0
        per_document_chars: dict[int | None, int] = {}

        for chunk in chunks:
            if len(selected) >= request.max_chunks:
                skipped_count += 1
                continue
            chunk_chars = len(format_context_chunk(chunk))
            separator_chars = 2 if selected else 0
            current_doc_chars = per_document_chars.get(chunk.document_id, 0)
            if total_chars + separator_chars + chunk_chars > request.max_context_chars:
                skipped_count += 1
                continue
            if current_doc_chars + chunk_chars > request.max_chars_per_document:
                skipped_count += 1
                continue
            selected.append(chunk)
            total_chars += separator_chars + chunk_chars
            per_document_chars[chunk.document_id] = current_doc_chars + chunk_chars

        return selected, skipped_count, skipped_count > 0

    def _simple_select(
        self,
        request: ContextBuildRequest,
    ) -> tuple[list[RerankedChunk], int, bool]:
        selected: list[RerankedChunk] = []
        seen_ids: set[str] = set()
        total_chars = 0
        skipped_count = 0

        for chunk in sorted(
            request.chunks,
            key=lambda item: item.rerank_score,
            reverse=True,
        ):
            if chunk.id in seen_ids:
                skipped_count += 1
                continue
            if len(selected) >= request.max_chunks:
                skipped_count += 1
                continue
            chunk_chars = len(format_context_chunk(chunk))
            separator_chars = 2 if selected else 0
            if total_chars + separator_chars + chunk_chars > request.max_context_chars:
                skipped_count += 1
                continue
            seen_ids.add(chunk.id)
            selected.append(chunk)
            total_chars += separator_chars + chunk_chars

        return selected, skipped_count, skipped_count > 0

    def _apply_multi_document_quota(
        self,
        chunks: list[RerankedChunk],
        *,
        target_document_ids: list[int],
        max_per_document: int,
        min_documents: int,
    ) -> list[RerankedChunk]:
        if max_per_document <= 0:
            return chunks
        selected: list[RerankedChunk] = []
        rejected: list[RerankedChunk] = []
        counts: dict[int | None, int] = {}

        selected_ids: set[str] = set()
        for document_id in target_document_ids:
            first = next(
                (chunk for chunk in chunks if chunk.document_id == document_id),
                None,
            )
            if first is not None and first.id not in selected_ids:
                selected.append(first)
                selected_ids.add(first.id)
                counts[first.document_id] = counts.get(first.document_id, 0) + 1

        for chunk in chunks:
            if chunk.id in selected_ids:
                continue
            count = counts.get(chunk.document_id, 0)
            if count < max_per_document:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                counts[chunk.document_id] = count + 1
            else:
                rejected.append(chunk)

        candidate_doc_count = len(
            {chunk.document_id for chunk in chunks if chunk.document_id is not None}
        )
        if candidate_doc_count < min_documents:
            return chunks
        return [*selected, *rejected]

    def _restore_document_order(
        self,
        chunks: list[RerankedChunk],
        target_document_ids: list[int],
    ) -> list[RerankedChunk]:
        groups: dict[int | None, list[RerankedChunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.document_id, []).append(chunk)

        target_set = set(target_document_ids)
        document_order = [
            *[doc_id for doc_id in target_document_ids if doc_id in groups],
            *[doc_id for doc_id in groups if doc_id not in target_set],
        ]
        ordered: list[RerankedChunk] = []
        for document_id in document_order:
            ordered.extend(sorted(groups[document_id], key=self._structure_key))
        return ordered

    def _chunk_identity(self, chunk: RerankedChunk) -> tuple:
        if chunk.document_id is not None and chunk.chunk_index is not None:
            return ("position", chunk.document_id, chunk.chunk_index)
        return ("id", chunk.id)

    def _choose_better(self, left: RerankedChunk, right: RerankedChunk) -> RerankedChunk:
        if right.rerank_score > left.rerank_score:
            winner, loser = right, left
        else:
            winner, loser = left, right
        metadata = self._merge_metadata(winner.metadata, loser.metadata)
        return winner.model_copy(update={"metadata": metadata})

    def _merge_chunks(self, left: RerankedChunk, right: RerankedChunk) -> RerankedChunk:
        winner = self._choose_better(left, right)
        loser = right if winner.id == left.id else left
        text = self._merge_text(winner.text, loser.text)
        metadata = self._merge_metadata(winner.metadata, loser.metadata)
        return winner.model_copy(update={"text": text, "metadata": metadata})

    def _merge_text(self, left: str, right: str) -> str:
        if right in left:
            return left
        if left in right:
            return right
        return f"{left}\n{right}"

    def _merge_metadata(self, primary: dict, secondary: dict) -> dict:
        metadata = {**secondary, **primary}
        citations = []
        for item in (secondary.get("citations"), primary.get("citations")):
            if isinstance(item, list):
                citations.extend(item)
        if citations:
            metadata["citations"] = list(dict.fromkeys(str(item) for item in citations))
        return metadata

    def _overlap_index(
        self,
        chunks: list[RerankedChunk],
        chunk: RerankedChunk,
    ) -> int | None:
        normalized_text = self._normalize_text(chunk.text)
        for index, existing in enumerate(chunks):
            if existing.document_id != chunk.document_id:
                continue
            similarity = self._similarity(
                normalized_text,
                self._normalize_text(existing.text),
            )
            if similarity >= self.overlap_threshold:
                return index
        return None

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left in right or right in left:
            return 1.0
        left_tokens = set(left)
        right_tokens = set(right)
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return overlap / union if union else 0.0

    def _normalize_text(self, text: str) -> str:
        return "".join(text.lower().split())

    def _priority_key(self, chunk: RerankedChunk) -> tuple:
        metadata = chunk.metadata
        return (
            bool(metadata.get("structure_match")),
            bool(metadata.get("identity_match")),
            chunk.rerank_score,
            metadata.get("original_fusion_score", chunk.original_score),
        )

    def _structure_key(self, chunk: RerankedChunk) -> tuple:
        metadata = chunk.metadata
        return (
            self._section_path_key(metadata.get("section_path")),
            self._number(metadata.get("chapter_number")),
            self._number(metadata.get("article_number")),
            self._number(metadata.get("page_start", metadata.get("page"))),
            chunk.chunk_index if chunk.chunk_index is not None else 10**9,
        )

    def _section_path_key(self, value) -> str:
        if isinstance(value, list):
            return " > ".join(str(item) for item in value)
        return str(value or "")

    def _number(self, value) -> int:
        return value if isinstance(value, int) else 10**9

    def _to_context_chunk(self, chunk: RerankedChunk) -> ContextChunk:
        return ContextChunk(
            id=chunk.id,
            text=chunk.text,
            document_id=chunk.document_id,
            knowledge_base_id=chunk.knowledge_base_id,
            chunk_index=chunk.chunk_index,
            score=chunk.rerank_score,
            source=chunk.metadata.get("source"),
            metadata=chunk.metadata,
        )

    def _citation(self, chunk: ContextChunk) -> dict:
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "chunk_index": chunk.chunk_index,
            "source": chunk.source,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }

    def _document_groups(self, chunks: list[ContextChunk]) -> list[dict]:
        groups: dict[int | None, list[ContextChunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.document_id, []).append(chunk)
        return [
            {
                "document_id": document_id,
                "chunk_ids": [chunk.id for chunk in group_chunks],
                "chunk_count": len(group_chunks),
                "total_chars": sum(len(chunk.text) for chunk in group_chunks),
            }
            for document_id, group_chunks in groups.items()
        ]

    def _metadata(
        self,
        *,
        request: ContextBuildRequest,
        started: float,
        input_count: int,
        output_count: int,
        document_groups: list[dict],
        deduplicated_count: int,
        merged_count: int,
        skipped_count: int,
        truncated: bool,
        failed: bool,
        error: str | None,
    ) -> dict:
        return {
            "enabled": True,
            "route_type": request.route_type,
            "strategy": request.strategy,
            "input_count": input_count,
            "output_count": output_count,
            "document_count": len(document_groups),
            "selected_document_ids": [
                group["document_id"] for group in document_groups
            ],
            "deduplicated_count": deduplicated_count,
            "merged_count": merged_count,
            "skipped_count": skipped_count,
            "truncated": truncated,
            "max_chars": request.max_context_chars,
            "max_chunks": request.max_chunks,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "failed": failed,
            "error": error,
        }
