from typing import Any

from backend.app.embeddings.base import BaseEmbedding
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from openai import OpenAI


class DashScopeEmbedding(BaseEmbedding):
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    provider_name = "dashscope"

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model_name = config.model
        self.dimension = config.dimension or 0

        if not config.api_key:
            raise BusinessException(50011, "Embedding API Key未配置")

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "base_url": config.base_url or self.default_base_url,
        }
        self.client = OpenAI(**client_kwargs)

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            vectors_by_index: dict[int, list[float]] = {}
            for position, item in enumerate(response.data):
                item_index = getattr(item, "index", position)
                embedding = getattr(item, "embedding", [])
                vectors_by_index[item_index] = [float(value) for value in embedding]

            vectors = [vectors_by_index[index] for index in range(len(texts))]
            if vectors:
                self.dimension = len(vectors[0])
            return vectors
        except BusinessException:
            raise
        except Exception as exc:
            logger.exception("DashScope embedding call failed")
            raise BusinessException(50011, "Embedding模型调用失败") from exc

    def extra_metadata(self) -> dict:
        return {
            "dimension_change_notice": (
                "If embedding dimension changes, rebuild the Qdrant collection."
            ),
        }
