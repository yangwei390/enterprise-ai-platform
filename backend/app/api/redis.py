from backend.app.cache import get_redis_client
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter

router = APIRouter()


@router.get("/redis/health", response_model=ApiResponse)
def redis_health_check() -> ApiResponse:
    logger.info("Redis health check requested")
    try:
        redis_client = get_redis_client()
        redis_client.ping()
    except Exception as exc:
        logger.warning(f"Redis health check failed: {exc}")
        raise BusinessException(50002, "Redis连接失败") from exc

    logger.info("Redis health check succeeded")
    return success(data={"status": "connected"})
