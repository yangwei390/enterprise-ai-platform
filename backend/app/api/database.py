from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success


router = APIRouter()


@router.get("/db/health", response_model=ApiResponse)
def database_health_check(db: Session = Depends(get_db)) -> ApiResponse:
    logger.info("Database health check requested")
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning(f"Database health check failed: {exc}")
        raise BusinessException(50001, "数据库连接失败") from exc

    logger.info("Database health check succeeded")
    return success(data={"status": "connected"})
