import re

from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.hybrid.base import HybridRetrieveQuery, HybridRetrieveResult
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.hybrid.fusion import rrf_fusion
from backend.app.retrievers.hybrid.sparse_retriever import BM25SparseRetriever
from backend.app.retrievers.metadata_filter import AutoMetadataFilterBuilder


class HybridRetriever:
    sparse_intent_patterns = (
        re.compile(r"第[0-9一二三四五六七八九十百千万几]+[章节条]"),
        re.compile(r"(章节|条款|法律|法规|劳动法)"),
    )

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()

    def retrieve(self, query: HybridRetrieveQuery) -> HybridRetrieveResult:
        dense_chunks = self.dense_retriever.retrieve(query)
        sparse_chunks = self.sparse_retriever.retrieve(query)
        auto_filter_result = AutoMetadataFilterBuilder().build(
            query=query.query,
            knowledge_base_id=query.knowledge_base_id,
            metadata_filter=query.metadata_filter,
        )
        retrieval_intent = self._detect_retrieval_intent(query.query)
        sparse_boosted = retrieval_intent == "sparse"

        if sparse_boosted:
            fused_chunks = self._sparse_first_fusion(
                dense_chunks=dense_chunks,
                sparse_chunks=sparse_chunks,
                top_k=query.top_k,
            )
            fusion_strategy = "sparse_first"
        else:
            fused_chunks = rrf_fusion(
                dense_chunks=dense_chunks,
                sparse_chunks=sparse_chunks,
                top_k=query.top_k,
            )
            fusion_strategy = "rrf"

        fused_chunks, soft_boost_applied = self._apply_soft_boost(
            chunks=fused_chunks,
            candidate_document_ids=auto_filter_result.candidate_document_ids,
            top_k=query.top_k,
        )

        return HybridRetrieveResult(
            chunks=fused_chunks,
            total=len(fused_chunks),
            metadata={
                "retriever_mode": "hybrid",
                "retrieval_intent": retrieval_intent,
                "sparse_boosted": sparse_boosted,
                "dense_total": len(dense_chunks),
                "sparse_total": len(sparse_chunks),
                "fused_total": len(fused_chunks),
                "fusion": fusion_strategy,
                "bm25_enabled": True,
                "auto_filter_applied": auto_filter_result.auto_filter_applied,
                "candidate_document_ids": auto_filter_result.candidate_document_ids,
                "source_hints": auto_filter_result.source_hints,
                "soft_boost_enabled": auto_filter_result.soft_boost_enabled,
                "soft_boost_applied": soft_boost_applied,
                "soft_boost_factor": 1.2,
                "auto_filter": auto_filter_result.metadata,
            },
        )

    def _detect_retrieval_intent(self, query: str) -> str:
        normalized_query = query.strip()
        if any(pattern.search(normalized_query) for pattern in self.sparse_intent_patterns):
            return "sparse"
        return "hybrid"

    def _sparse_first_fusion(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        fused_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        dense_ranks = {chunk.id: rank for rank, chunk in enumerate(dense_chunks, start=1)}

        for sparse_rank, chunk in enumerate(sparse_chunks, start=1):
            if chunk.id in seen_ids:
                continue
            fusion_score = 1.0 + 1 / sparse_rank
            metadata = {
                **chunk.metadata,
                "fusion_strategy": "sparse_first",
                "sparse_boosted": True,
                "dense_rank": dense_ranks.get(chunk.id),
                "sparse_rank": sparse_rank,
                "fusion_score": fusion_score,
                "sparse_score": chunk.metadata.get("sparse_score", chunk.score),
            }
            fused_chunks.append(
                chunk.model_copy(update={"score": fusion_score, "metadata": metadata})
            )
            seen_ids.add(chunk.id)
            if len(fused_chunks) >= top_k:
                return fused_chunks

        for dense_rank, chunk in enumerate(dense_chunks, start=1):
            if chunk.id in seen_ids:
                continue
            fusion_score = 1 / (100 + dense_rank)
            metadata = {
                **chunk.metadata,
                "fusion_strategy": "sparse_first_dense_backfill",
                "sparse_boosted": True,
                "dense_rank": dense_rank,
                "sparse_rank": None,
                "fusion_score": fusion_score,
            }
            fused_chunks.append(
                chunk.model_copy(update={"score": fusion_score, "metadata": metadata})
            )
            seen_ids.add(chunk.id)
            if len(fused_chunks) >= top_k:
                break

        return fused_chunks

    def _apply_soft_boost(
        self,
        chunks: list[RetrievedChunk],
        candidate_document_ids: list[int],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], bool]:
        if not candidate_document_ids:
            return chunks, False

        candidate_ids = set(candidate_document_ids)
        boosted_chunks: list[RetrievedChunk] = []
        soft_boost_applied = False
        for chunk in chunks:
            if chunk.document_id not in candidate_ids:
                boosted_chunks.append(chunk)
                continue

            boosted_score = chunk.score * 1.2
            metadata = {
                **chunk.metadata,
                "auto_filter_candidate": True,
                "soft_boost_applied": True,
                "soft_boost_factor": 1.2,
                "score_before_soft_boost": chunk.score,
                "fusion_score": boosted_score,
            }
            boosted_chunks.append(
                chunk.model_copy(update={"score": boosted_score, "metadata": metadata})
            )
            soft_boost_applied = True

        boosted_chunks.sort(key=lambda item: item.score, reverse=True)
        return boosted_chunks[:top_k], soft_boost_applied
