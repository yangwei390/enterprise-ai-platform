from fastapi import FastAPI

from backend.app.api import health_router
from backend.app.config.settings import settings
from backend.app.exceptions import register_exception_handlers
from backend.app.logger import logger, setup_logger


def create_app() -> FastAPI:
    setup_logger()

    app = FastAPI(title=settings.APP_NAME)
    logger.info("Enterprise AI Platform started")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.APP_VERSION}")

    register_exception_handlers(app)
    app.include_router(health_router)

    return app


app = create_app()
