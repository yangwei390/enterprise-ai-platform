from typing import Any

from pydantic import BaseModel


class ApiResponse(BaseModel):
    code: int
    message: str
    data: Any | None


def success(data: Any | None = None, message: str = "success") -> ApiResponse:
    return ApiResponse(code=0, message=message, data=data)


def error(
    message: str = "error",
    code: int = 1,
    data: Any | None = None,
) -> ApiResponse:
    return ApiResponse(code=code, message=message, data=data)
