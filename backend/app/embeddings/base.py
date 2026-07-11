from abc import ABC, abstractmethod
from math import ceil

from backend.app.chunkers import Chunk
from backend.app.embeddings.config import get_embedding_config
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from pydantic import BaseModel, Field


class EmbeddingItem(BaseModel):
    chunk_index: int
    text: str
    vector: list[float]
    document_id: int | None
    knowledge_base_id: int | None
    metadata: dict = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    items: list[EmbeddingItem]
    total_items: int
    model_name: str
    dimension: int
    metadata: dict = Field(default_factory=dict)


class EmbeddingBatchError(BusinessException):
    def __init__(self, message: str, metadata: dict) -> None:
        self.metadata = metadata
        super().__init__(50011, message)


class BaseEmbedding(ABC):
    model_name: str
    dimension: int
    provider_name: str = "base"

    def embed_text(self, text: str) -> list[float]:
        vectors, _ = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        if not texts:
            return [], {
                "total_texts": 0,
                "batch_size": self._batch_size(),
                "batch_count": 0,
                "failed_batch": None,
            }

        batch_size = self._batch_size()
        batch_count = ceil(len(texts) / batch_size)
        vectors: list[list[float]] = []
        metadata = {
            "total_texts": len(texts),
            "batch_size": batch_size,
            "batch_count": batch_count,
            "failed_batch": None,
        }

        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            logger.info(
                "Embedding batch started | total_texts=%s | batch_size=%s | "
                "batch_count=%s | current_batch=%s",
                len(texts),
                batch_size,
                batch_count,
                batch_index,
            )
            try:
                batch_vectors = self.embed_text_batch(batch)
            except Exception as exc:
                metadata["failed_batch"] = batch_index
                logger.exception(
                    "Embedding batch failed | total_texts=%s | batch_size=%s | "
                    "batch_count=%s | failed_batch=%s",
                    len(texts),
                    batch_size,
                    batch_count,
                    batch_index,
                )
                raise EmbeddingBatchError("Embedding批次调用失败", metadata) from exc

            if len(batch_vectors) != len(batch):
                metadata["failed_batch"] = batch_index
                raise EmbeddingBatchError("Embedding返回数量与输入数量不一致", metadata)
            vectors.extend(batch_vectors)

        if vectors:
            self.dimension = len(vectors[0])
        return vectors, metadata

    @abstractmethod
    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingResult:
        texts = [chunk.text for chunk in chunks]
        vectors, batch_metadata = self.embed_texts(texts)
        items = [
            EmbeddingItem(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=vectors[index],
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
                metadata=chunk.metadata,
            )
            for index, chunk in enumerate(chunks)
        ]
        dimension = len(vectors[0]) if vectors else self.dimension
        self.dimension = dimension
        return EmbeddingResult(
            items=items,
            total_items=len(items),
            model_name=self.model_name,
            dimension=dimension,
            metadata={
                "embedding_provider": self.provider_name,
                "embedding_model": self.model_name,
                "embedding_dimension": dimension,
                **batch_metadata,
                **self.extra_metadata(),
            },
        )

    def extra_metadata(self) -> dict:
        return {}

    def _batch_size(self) -> int:
        config = getattr(self, "config", None)
        batch_size = getattr(config, "batch_size", None) or get_embedding_config().batch_size
        return max(1, batch_size)
