from backend.app.api.health import router as health_router
from backend.app.api.database import router as database_router

__all__ = ["database_router", "health_router"]
