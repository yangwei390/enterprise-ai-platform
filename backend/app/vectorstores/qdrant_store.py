import hashlib
import uuid
from typing import Any, cast

from backend.app.exceptions import BusinessException
from backend.app.vector.qdrant_client import get_qdrant_client
from backend.app.vectorstores.base import BaseVectorStore, VectorRecord, VectorStoreResult
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantVectorStore(BaseVectorStore):
    collection_name = "enterprise_ai_chunks"
    max_payload_text_length = 2000

    def upsert(self, records: list[VectorRecord]) -> VectorStoreResult:
        if not records:
            return VectorStoreResult(
                collection_name=self.collection_name,
                total_records=0,
                ids=[],
                metadata={"vector_store": "qdrant"},
            )

        vector_size = len(records[0].vector)
        client = get_qdrant_client()
        self._ensure_collection(vector_size)

        point_ids = [self._build_point_id(record) for record in records]
        points = [
            PointStruct(
                id=point_id,
                vector=record.vector,
                payload=self._build_payload(record),
            )
            for record, point_id in zip(records, point_ids, strict=True)
        ]
        client.upsert(collection_name=self.collection_name, points=points)

        return VectorStoreResult(
            collection_name=self.collection_name,
            total_records=len(records),
            ids=point_ids,
            metadata={
                "vector_store": "qdrant",
                "vector_size": vector_size,
            },
        )

    def _ensure_collection(self, vector_size: int) -> None:
        client = get_qdrant_client()
        if client.collection_exists(self.collection_name):
            existing_vector_size = self._get_collection_vector_size()
            if existing_vector_size is not None and existing_vector_size != vector_size:
                raise BusinessException(
                    50005,
                    (
                        "Qdrant collection向量维度不匹配，"
                        f"当前collection维度={existing_vector_size}，"
                        f"新Embedding维度={vector_size}，请重建collection后重新索引"
                    ),
                )
            return

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _get_collection_vector_size(self) -> int | None:
        client = get_qdrant_client()
        collection_info = client.get_collection(self.collection_name)
        vectors = collection_info.config.params.vectors
        vectors_any = cast(Any, vectors)
        if hasattr(vectors_any, "size"):
            size = vectors_any.size
            return size if isinstance(size, int) else None
        if isinstance(vectors, dict):
            first_vector = next(iter(vectors.values()), None)
            if first_vector is None:
                return None
            return self._get_vector_size_from_config(first_vector)
        return self._get_vector_size_from_config(vectors)

    def _get_vector_size_from_config(self, vector_config: Any) -> int | None:
        size = getattr(vector_config, "size", None)
        return size if isinstance(size, int) else None

    def _build_point_id(self, record: VectorRecord) -> str:
        business_key = self._build_business_key(record)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, business_key))

    def _build_business_key(self, record: VectorRecord) -> str:
        document_id = record.document_id if record.document_id is not None else "unknown"
        chunk_index = record.chunk_index if record.chunk_index is not None else 0
        text_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:8]
        return f"{document_id}_{chunk_index}_{text_hash}"

    def _build_payload(self, record: VectorRecord) -> dict:
        original_text_length = len(record.text)
        text_truncated = original_text_length > self.max_payload_text_length
        payload_text = record.text[: self.max_payload_text_length]
        metadata = {
            **record.metadata,
            "text_truncated": text_truncated,
            "original_text_length": original_text_length,
        }

        return {
            "business_key": self._build_business_key(record),
            "text": payload_text,
            "document_id": record.document_id,
            "knowledge_base_id": record.knowledge_base_id,
            "chunk_index": record.chunk_index,
            "source": metadata.get("source"),
            "page_count": metadata.get("page_count"),
            "parser": metadata.get("parser"),
            "cleaner": metadata.get("cleaner"),
            "strategy": metadata.get("strategy"),
            "chunk_size": metadata.get("chunk_size"),
            "chunk_overlap": metadata.get("chunk_overlap"),
            "start_offset": metadata.get("start_offset"),
            "end_offset": metadata.get("end_offset"),
            "token_count": metadata.get("token_count"),
            "metadata": metadata,
        }
