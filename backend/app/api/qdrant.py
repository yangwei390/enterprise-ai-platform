from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success
from backend.app.vector import get_qdrant_client
from fastapi import APIRouter

router = APIRouter()


@router.get("/qdrant/health", response_model=ApiResponse)
def qdrant_health_check() -> ApiResponse:
    logger.info("Qdrant health check requested")
    try:
        qdrant_client = get_qdrant_client()
        qdrant_client.get_collections()
    except Exception as exc:
        logger.warning(f"Qdrant health check failed: {exc}")
        raise BusinessException(50003, "Qdrant连接失败") from exc

    logger.info("Qdrant health check succeeded")
    return success(data={"status": "connected"})
