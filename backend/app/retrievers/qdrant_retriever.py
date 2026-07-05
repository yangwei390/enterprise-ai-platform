from qdrant_client.http.api_client import jsonable_encoder
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchRequest

from backend.app.embeddings import EmbeddingFactory
from backend.app.vector.qdrant_client import get_qdrant_client
from backend.app.vectorstores import QdrantVectorStore
from backend.app.retrievers.base import BaseRetriever, RetrievedChunk, RetrieveQuery, RetrieveResult


class QdrantRetriever(BaseRetriever):
    collection_name = QdrantVectorStore.collection_name

    def retrieve(self, query: RetrieveQuery) -> RetrieveResult:
        embedding = EmbeddingFactory.get_embedding()
        query_vector = embedding.embed_text(query.query)
        query_filter = self._build_filter(query.knowledge_base_id)

        points = self._search_points(
            query_vector=query_vector,
            query_filter=query_filter,
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

    def _build_filter(self, knowledge_base_id: int | None) -> Filter | None:
        if knowledge_base_id is None:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="knowledge_base_id",
                    match=MatchValue(value=knowledge_base_id),
                )
            ]
        )

    def _to_retrieved_chunk(self, point) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            id=str(point.id),
            score=float(point.score),
            text=payload.get("text", ""),
            document_id=payload.get("document_id"),
            knowledge_base_id=payload.get("knowledge_base_id"),
            chunk_index=payload.get("chunk_index"),
            metadata=payload.get("metadata") or {},
        )

    def _search_points(
        self,
        query_vector: list[float],
        query_filter: Filter | None,
        limit: int,
    ) -> list:
        search_request = SearchRequest(
            vector=query_vector,
            filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        response = get_qdrant_client().http.points_api.api_client.request(
            type_=object,
            method="POST",
            url="/collections/{collection_name}/points/search",
            path_params={"collection_name": self.collection_name},
            headers={"Content-Type": "application/json"},
            content=jsonable_encoder(search_request),
        )

        if isinstance(response, dict):
            return response.get("result", [])

        return getattr(response, "result", [])
