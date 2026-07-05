from backend.app.rerankers.base import BaseReranker, RerankedChunk, RerankQuery, RerankResult


class DummyReranker(BaseReranker):
    def rerank(self, query: RerankQuery) -> RerankResult:
        sorted_chunks = sorted(
            query.chunks,
            key=lambda chunk: chunk.score,
            reverse=True,
        )[: query.top_k]

        chunks = [
            RerankedChunk(
                id=chunk.id,
                original_score=chunk.score,
                rerank_score=chunk.score,
                text=chunk.text,
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
            for chunk in sorted_chunks
        ]

        return RerankResult(
            query=query.query,
            top_k=query.top_k,
            total=len(chunks),
            chunks=chunks,
            metadata={
                "reranker": "dummy",
                "strategy": "score_desc",
            },
        )
