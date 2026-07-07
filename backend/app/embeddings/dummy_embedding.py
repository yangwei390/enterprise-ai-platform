import hashlib

from backend.app.chunkers import Chunk
from backend.app.embeddings.base import BaseEmbedding, EmbeddingItem, EmbeddingResult
from backend.app.embeddings.config import get_embedding_config


class DummyEmbedding(BaseEmbedding):
    default_dimension = 8
    model_name = "dummy-embedding"
    config = get_embedding_config()
    dimension = config.dimension or default_dimension

    def embed_text(self, text: str) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")
        counter = 0

        while len(values) < self.dimension:
            digest = hashlib.sha256(seed + str(counter).encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                if len(values) >= self.dimension:
                    break
                values.append(
                    int.from_bytes(digest[index : index + 4], byteorder="big")
                    / 0xFFFFFFFF
                )
            counter += 1

        return values

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
