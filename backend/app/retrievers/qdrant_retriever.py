import requests

from backend.app.config.settings import settings
from backend.app.embeddings import EmbeddingFactory
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.vectorstores import QdrantVectorStore
from backend.app.retrievers.base import BaseRetriever, RetrievedChunk, RetrieveQuery, RetrieveResult


class QdrantRetriever(BaseRetriever):
    collection_name = QdrantVectorStore.collection_name

    def retrieve(self, query: RetrieveQuery) -> RetrieveResult:
        embedding = EmbeddingFactory.get_embedding()
        query_vector = embedding.embed_text(query.query)

        points = self._search_points(
            query_vector=query_vector,
            knowledge_base_id=query.knowledge_base_id,
            limit=query.top_k,
        )
        chunks = [self._to_retrieved_chunk(point) for point in points]

        return RetrieveResult(
            query=query.query,
            top_k=query.top_k,
            total=len(chunks),
            chunks=chunks,
            metadata={
                "vector_store": "qdrant",
                "collection_name": self.collection_name,
                "embedding_model": embedding.model_name,
            },
        )

    def _build_filter(self, knowledge_base_id: int | None) -> dict | None:
        if knowledge_base_id is None:
            return None

        return {
            "must": [
                {
                    "key": "knowledge_base_id",
                    "match": {
                        "value": knowledge_base_id,
                    },
                }
            ]
        }

    def _to_retrieved_chunk(self, point) -> RetrievedChunk:
        if isinstance(point, dict):
            payload = point.get("payload") or {}
            point_id = point.get("id")
            score = point.get("score", 0.0)
        else:
            payload = point.payload or {}
            point_id = point.id
            score = point.score

        return RetrievedChunk(
            id=str(point_id),
            score=float(score),
            text=payload.get("text", ""),
            document_id=payload.get("document_id"),
            knowledge_base_id=payload.get("knowledge_base_id"),
            chunk_index=payload.get("chunk_index"),
            metadata=payload.get("metadata") or {},
        )

    def _search_points(
        self,
        query_vector: list[float],
        knowledge_base_id: int | None,
        limit: int,
    ) -> list:
        body = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        query_filter = self._build_filter(knowledge_base_id)
        if query_filter is not None:
            body["filter"] = query_filter

        url = (
            f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            f"/collections/{self.collection_name}/points/search"
        )
        response = requests.post(url, json=body, timeout=30)
        if response.status_code != 200:
            logger.error(
                f"Qdrant search failed | status_code={response.status_code} | "
                f"response={response.text}"
            )
            raise BusinessException(50004, "向量检索失败")

        return response.json().get("result", [])
