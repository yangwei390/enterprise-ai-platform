from fastapi import FastAPI

from backend.app.config.settings import ConfigResponse, settings
from backend.app.logger import logger, setup_logger


def create_app() -> FastAPI:
    setup_logger()

    app = FastAPI(title=settings.APP_NAME)
    logger.info("Enterprise AI Platform started")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.APP_VERSION}")

    @app.get("/health")
    def health_check():
        logger.info("Health check requested")
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        }

    @app.get("/config", response_model=ConfigResponse)
    def get_config() -> ConfigResponse:
        logger.info("Config requested")
        return ConfigResponse(
            app_name=settings.APP_NAME,
            app_env=settings.APP_ENV,
            app_version=settings.APP_VERSION,
            app_host=settings.APP_HOST,
            app_port=settings.APP_PORT,
        )

    return app


app = create_app()
