from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from backend.app.exceptions.custom import BusinessException
from backend.app.logger import logger


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BusinessException)
    async def business_exception_handler(_, exc: BusinessException):
        logger.warning(exc.message)
        return JSONResponse(
            status_code=200,
            content={
                "code": exc.code,
                "message": exc.message,
                "data": None,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.status_code,
                "message": exc.detail,
                "data": None,
            },
        )

    @app.exception_handler(Exception)
    async def unknown_exception_handler(_, exc: Exception):
        logger.exception(exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": 50000,
                "message": "服务器内部错误",
                "data": None,
            },
        )
