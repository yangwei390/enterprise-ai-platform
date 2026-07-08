import requests
from backend.app.config.settings import settings
from backend.app.rerankers import RerankedChunk
from backend.app.vectorstores import QdrantVectorStore


class QdrantNeighborChunkLookup:
    collection_name = QdrantVectorStore.collection_name

    def find_neighbor(
        self,
        *,
        document_id: int,
        knowledge_base_id: int,
        chunk_index: int,
    ) -> RerankedChunk | None:
        body = {
            "filter": {
                "must": [
                    {
                        "key": "document_id",
                        "match": {
                            "value": document_id,
                        },
                    },
                    {
                        "key": "knowledge_base_id",
                        "match": {
                            "value": knowledge_base_id,
                        },
                    },
                    {
                        "key": "chunk_index",
                        "match": {
                            "value": chunk_index,
                        },
                    },
                ]
            },
            "limit": 1,
            "with_payload": True,
            "with_vector": False,
        }
        url = (
            f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            f"/collections/{self.collection_name}/points/scroll"
        )
        response = requests.post(url, json=body, timeout=30)
        response.raise_for_status()
        result = response.json().get("result") or {}
        points = result.get("points") or []
        if not points:
            return None
        return self._to_reranked_chunk(points[0])

    def _to_reranked_chunk(self, point: dict) -> RerankedChunk:
        payload = point.get("payload") or {}
        metadata = {
            **(payload.get("metadata") or {}),
            "source": payload.get("source") or (payload.get("metadata") or {}).get("source"),
            "document_id": payload.get("document_id"),
            "knowledge_base_id": payload.get("knowledge_base_id"),
            "chunk_index": payload.get("chunk_index"),
            "parser": payload.get("parser"),
            "cleaner": payload.get("cleaner"),
            "strategy": payload.get("strategy"),
        }
        return RerankedChunk(
            id=str(point.get("id")),
            original_score=0.0,
            rerank_score=0.0,
            text=payload.get("text", ""),
            document_id=payload.get("document_id"),
            knowledge_base_id=payload.get("knowledge_base_id"),
            chunk_index=payload.get("chunk_index"),
            metadata=metadata,
        )
