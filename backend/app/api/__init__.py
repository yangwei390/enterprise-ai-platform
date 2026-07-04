from backend.app.api.database import router as database_router
from backend.app.api.document import router as document_router
from backend.app.api.health import router as health_router
from backend.app.api.knowledge_base import router as knowledge_base_router
from backend.app.api.qdrant import router as qdrant_router
from backend.app.api.redis import router as redis_router

__all__ = [
    "database_router",
    "document_router",
    "health_router",
    "knowledge_base_router",
    "qdrant_router",
    "redis_router",
]
