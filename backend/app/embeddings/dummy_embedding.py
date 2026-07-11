import hashlib

from backend.app.embeddings.base import BaseEmbedding
from backend.app.embeddings.config import get_embedding_config


class DummyEmbedding(BaseEmbedding):
    default_dimension = 8
    model_name = "dummy-embedding"
    provider_name = "dummy"
    config = get_embedding_config()
    dimension = config.dimension or default_dimension

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
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
