from backend.app.api import (
    agent_router,
    chat_router,
    conversation_router,
    database_router,
    debug_router,
    document_router,
    evaluation_router,
    health_router,
    knowledge_base_router,
    mcp_router,
    product_router,
    qdrant_router,
    redis_router,
    retriever_router,
    tools_router,
    workflow_router,
)
from backend.app.config.settings import settings
from backend.app.exceptions import register_exception_handlers
from backend.app.logger import logger, setup_logger
from backend.app.mcp.client_manager import get_mcp_client_manager
from backend.app.middleware import RequestLogMiddleware
from backend.app.tools import get_tool_registry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    setup_logger()

    app = FastAPI(title=settings.APP_NAME)
    logger.info("Enterprise AI Platform started")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.APP_VERSION}")

    register_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://192.168.0.121:5173",
            "http://192.168.0.111:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLogMiddleware)

    @app.on_event("startup")
    async def startup_mcp() -> None:
        if settings.MCP_ENABLED and settings.MCP_DISCOVERY_ON_STARTUP:
            await get_tool_registry().arefresh()

    @app.on_event("shutdown")
    async def shutdown_mcp() -> None:
        await get_mcp_client_manager().disconnect_all()

    app.include_router(agent_router)
    app.include_router(health_router)
    app.include_router(database_router)
    app.include_router(debug_router)
    app.include_router(redis_router)
    app.include_router(qdrant_router)
    app.include_router(mcp_router)
    app.include_router(product_router)
    app.include_router(knowledge_base_router)
    app.include_router(document_router)
    app.include_router(evaluation_router)
    app.include_router(retriever_router)
    app.include_router(tools_router)
    app.include_router(workflow_router)
    app.include_router(conversation_router)
    app.include_router(chat_router)

    return app


app = create_app()
