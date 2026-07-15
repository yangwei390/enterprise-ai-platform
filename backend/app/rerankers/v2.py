from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from backend.app.rerankers.base import RerankedChunk

if TYPE_CHECKING:
    from backend.app.retrievers.pipeline.context import RetrieverPipelineContext


@dataclass
class RerankerV2Result:
    chunks: list[RerankedChunk]
    metadata: dict


class RerankerV2Enhancer:
    structure_boost = 0.02
    identity_boost = 0.01

    def enhance(
        self,
        *,
        chunks: list[RerankedChunk],
        context: RetrieverPipelineContext,
        diversity_enabled: bool,
        max_per_document: int,
        min_documents: int,
    ) -> RerankerV2Result:
        started = perf_counter()
        route_type = (
            context.routing_result.route_type if context.routing_result is not None else None
        )
        strategy = (
            context.retrieval_strategy.strategy
            if context.retrieval_strategy is not None
            else None
        )
        annotated = [
            self._annotate_chunk(chunk, rank=index, context=context)
            for index, chunk in enumerate(chunks, start=1)
        ]
        structure_boost_count = len(
            [chunk for chunk in annotated if chunk.metadata.get("structure_match")]
        )
        identity_boost_count = len(
            [chunk for chunk in annotated if chunk.metadata.get("identity_match")]
        )

        if strategy == "DOCUMENT":
            ranked = sorted(
                annotated,
                key=lambda chunk: (
                    self._effective_score(chunk),
                    -int(chunk.metadata.get("rerank_rank", 0)),
                ),
                reverse=True,
            )
            diversity_applied = False
            rejected_by_quota_count = 0
            backfilled_count = 0
        elif strategy == "MULTI_DOCUMENT" and diversity_enabled:
            ranked, rejected_by_quota_count, backfilled_count = self._apply_diversity(
                annotated,
                max_per_document=max_per_document,
                min_documents=min_documents,
                top_k=context.top_k,
            )
            diversity_applied = True
        else:
            ranked = annotated
            diversity_applied = False
            rejected_by_quota_count = 0
            backfilled_count = 0

        final_chunks = [
            self._with_rank(chunk, rank=rank) for rank, chunk in enumerate(ranked, start=1)
        ]
        return RerankerV2Result(
            chunks=final_chunks,
            metadata={
                "enabled": True,
                "route_type": route_type,
                "strategy": strategy,
                "input_count": len(chunks),
                "model_ranked_count": len(chunks),
                "output_count": len(final_chunks),
                "diversity_applied": diversity_applied,
                "max_per_document": max_per_document,
                "selected_document_ids": list(
                    dict.fromkeys(
                        [
                            chunk.document_id
                            for chunk in final_chunks
                            if chunk.document_id is not None
                        ]
                    )
                ),
                "rejected_by_quota_count": rejected_by_quota_count,
                "backfilled_count": backfilled_count,
                "structure_boost_count": structure_boost_count,
                "identity_boost_count": identity_boost_count,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "failed": False,
                "error": None,
            },
        )

    def _annotate_chunk(
        self,
        chunk: RerankedChunk,
        *,
        rank: int,
        context: RetrieverPipelineContext,
    ) -> RerankedChunk:
        structure_match = self._structure_match(chunk, context)
        identity_match = self._identity_match(chunk, context)
        metadata = {
            **chunk.metadata,
            "rerank_score": chunk.rerank_score,
            "rerank_rank": rank,
            "original_fusion_score": chunk.original_score,
            "document_id": chunk.document_id,
            "structure_match": structure_match,
            "identity_match": identity_match,
            "route_type": (
                context.routing_result.route_type
                if context.routing_result is not None
                else None
            ),
            "strategy": (
                context.retrieval_strategy.strategy
                if context.retrieval_strategy is not None
                else None
            ),
        }
        return chunk.model_copy(update={"metadata": metadata})

    def _structure_match(
        self,
        chunk: RerankedChunk,
        context: RetrieverPipelineContext,
    ) -> bool:
        plan = context.retrieval_plan
        if plan is None:
            return False
        structure_constraints = [
            constraint
            for constraint in plan.constraints
            if constraint.applied
            and constraint.field not in {"document_id", "knowledge_base_id"}
        ]
        if not structure_constraints:
            return False
        return all(
            self._matches_constraint(
                actual=chunk.metadata.get(constraint.field),
                operator=constraint.operator,
                expected=constraint.value,
            )
            for constraint in structure_constraints
        )

    def _matches_constraint(self, *, actual, operator: str, expected) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "in":
            values = expected if isinstance(expected, list) else [expected]
            return actual in values
        if operator == "contains":
            if isinstance(actual, list):
                return any(str(expected) in str(item) for item in actual)
            return isinstance(actual, str) and str(expected) in actual
        if operator == "prefix":
            if isinstance(actual, list):
                return any(str(item).startswith(str(expected)) for item in actual)
            return isinstance(actual, str) and actual.startswith(str(expected))
        if operator == "range":
            if not isinstance(actual, int | float):
                return False
            if isinstance(expected, dict):
                lower = expected.get("gte", expected.get("min"))
                upper = expected.get("lte", expected.get("max"))
            elif isinstance(expected, list | tuple) and len(expected) == 2:
                lower, upper = expected
            else:
                return False
            if lower is not None and actual < lower:
                return False
            if upper is not None and actual > upper:
                return False
            return True
        return False

    def _identity_match(
        self,
        chunk: RerankedChunk,
        context: RetrieverPipelineContext,
    ) -> bool:
        if context.routing_result is None or chunk.document_id is None:
            return False
        return chunk.document_id in set(context.routing_result.target_document_ids)

    def _effective_score(self, chunk: RerankedChunk) -> float:
        score = chunk.rerank_score
        if chunk.metadata.get("structure_match"):
            score += self.structure_boost
        if chunk.metadata.get("identity_match"):
            score += self.identity_boost
        return score

    def _apply_diversity(
        self,
        chunks: list[RerankedChunk],
        *,
        max_per_document: int,
        min_documents: int,
        top_k: int,
    ) -> tuple[list[RerankedChunk], int, int]:
        if max_per_document <= 0:
            return chunks[:top_k], 0, 0
        selected: list[RerankedChunk] = []
        rejected: list[RerankedChunk] = []
        per_document_counts: dict[int | None, int] = {}

        for chunk in chunks:
            count = per_document_counts.get(chunk.document_id, 0)
            if count < max_per_document:
                selected.append(chunk)
                per_document_counts[chunk.document_id] = count + 1
            else:
                rejected.append(chunk)
            if len(selected) >= top_k:
                break

        rejected_by_quota_count = len(rejected)
        candidate_document_count = len(
            {chunk.document_id for chunk in chunks if chunk.document_id is not None}
        )
        selected_document_count = len(
            {chunk.document_id for chunk in selected if chunk.document_id is not None}
        )
        if candidate_document_count < min_documents:
            return chunks[:top_k], 0, 0

        backfilled_count = 0
        if len(selected) < min(top_k, len(chunks)):
            selected_ids = {chunk.id for chunk in selected}
            for chunk in chunks:
                if len(selected) >= top_k:
                    break
                if chunk.id in selected_ids:
                    continue
                selected.append(chunk)
                selected_ids.add(chunk.id)
                backfilled_count += 1

        if selected_document_count == 0 and chunks:
            return chunks[:top_k], 0, 0
        return selected[:top_k], rejected_by_quota_count, backfilled_count

    def _with_rank(self, chunk: RerankedChunk, *, rank: int) -> RerankedChunk:
        metadata = {**chunk.metadata, "rerank_rank": rank}
        return chunk.model_copy(update={"metadata": metadata})
