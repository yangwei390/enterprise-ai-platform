import hashlib

from backend.app.chunkers import Chunk
from backend.app.embeddings.base import BaseEmbedding, EmbeddingItem, EmbeddingResult


class DummyEmbedding(BaseEmbedding):
    model_name = "dummy-embedding"
    dimension = 8

    def embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            int.from_bytes(digest[index * 4 : (index + 1) * 4], "big") / 0xFFFFFFFF
            for index in range(self.dimension)
        ]

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingResult:
        items = [
            EmbeddingItem(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=self.embed_text(chunk.text),
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        return EmbeddingResult(
            items=items,
            total_items=len(items),
            model_name=self.model_name,
            dimension=self.dimension,
            metadata={
                "embedding_provider": "dummy",
                "embedding_model": self.model_name,
                "embedding_dimension": self.dimension,
            },
        )
