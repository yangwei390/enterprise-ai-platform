from backend.app.retrievers.base import RetrievedChunk


def rrf_fusion(
    dense_chunks: list[RetrievedChunk],
    sparse_chunks: list[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    chunk_by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        chunk_by_id.setdefault(chunk.id, chunk)
        dense_ranks[chunk.id] = rank
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank)

    for rank, chunk in enumerate(sparse_chunks, start=1):
        chunk_by_id.setdefault(chunk.id, chunk)
        sparse_ranks[chunk.id] = rank
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank)

    sorted_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    fused_chunks: list[RetrievedChunk] = []
    for chunk_id in sorted_ids:
        chunk = chunk_by_id[chunk_id]
        metadata = {
            **chunk.metadata,
            "fusion_score": scores[chunk_id],
            "dense_rank": dense_ranks.get(chunk_id),
            "sparse_rank": sparse_ranks.get(chunk_id),
        }
        fused_chunks.append(
            chunk.model_copy(update={"score": scores[chunk_id], "metadata": metadata})
        )

    return fused_chunks
