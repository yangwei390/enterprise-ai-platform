from backend.app.api import (
    chat_router,
    conversation_router,
    database_router,
    document_router,
    health_router,
    knowledge_base_router,
    qdrant_router,
    redis_router,
    retriever_router,
    tools_router,
    workflow_router,
)
from backend.app.config.settings import settings
from backend.app.exceptions import register_exception_handlers
from backend.app.logger import logger, setup_logger
from backend.app.middleware import RequestLogMiddleware
from fastapi import FastAPI


def create_app() -> FastAPI:
    setup_logger()

    app = FastAPI(title=settings.APP_NAME)
    logger.info("Enterprise AI Platform started")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.APP_VERSION}")

    register_exception_handlers(app)
    app.add_middleware(RequestLogMiddleware)
    app.include_router(health_router)
    app.include_router(database_router)
    app.include_router(redis_router)
    app.include_router(qdrant_router)
    app.include_router(knowledge_base_router)
    app.include_router(document_router)
    app.include_router(retriever_router)
    app.include_router(tools_router)
    app.include_router(workflow_router)
    app.include_router(conversation_router)
    app.include_router(chat_router)

    return app


app = create_app()
