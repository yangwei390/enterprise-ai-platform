from fastapi import APIRouter

from backend.app.config.settings import ConfigResponse, settings
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success


router = APIRouter()


@router.get("/health", response_model=ApiResponse)
def health_check() -> ApiResponse:
    logger.info("Health check requested")
    return success(
        data={
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        }
    )


@router.get("/config", response_model=ApiResponse)
def get_config() -> ApiResponse:
    logger.info("Config requested")
    return success(
        data=ConfigResponse(
            app_name=settings.APP_NAME,
            app_env=settings.APP_ENV,
            app_version=settings.APP_VERSION,
            app_host=settings.APP_HOST,
            app_port=settings.APP_PORT,
        )
    )
