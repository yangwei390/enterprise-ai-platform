import hashlib
import uuid

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
            return

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

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
