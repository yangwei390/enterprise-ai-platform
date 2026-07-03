from qdrant_client import QdrantClient

from backend.app.config.settings import settings


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    return _qdrant_client


def close_qdrant_client() -> None:
    global _qdrant_client

    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None
