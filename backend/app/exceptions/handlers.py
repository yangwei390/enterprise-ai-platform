from backend.app.exceptions.custom import BusinessException
from backend.app.logger import logger
from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


def _status_code_from_business_code(code: int) -> int:
    if 40000 <= code < 40100:
        return 400
    if 40100 <= code < 40200:
        return 401
    if 40300 <= code < 40400:
        return 403
    if 40400 <= code < 40500:
        return 404
    if 41000 <= code < 41100:
        return 500
    if 50000 <= code < 60000:
        return 500
    return 400


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BusinessException)
    async def business_exception_handler(_, exc: BusinessException):
        logger.warning(exc.message)
        return JSONResponse(
            status_code=_status_code_from_business_code(exc.code),
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
