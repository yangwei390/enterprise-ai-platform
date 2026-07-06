from datetime import datetime

import requests
from backend.app.mcp.base import MCPToolConfig
from backend.app.tools.base import BaseTool, ToolResult
from pydantic import BaseModel, ConfigDict


class RemoteToolArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class RemoteHTTPTool(BaseTool):
    args_schema: type[BaseModel] = RemoteToolArgs
    source: str = "mcp"
    permission: str = "public"

    def __init__(self, config: MCPToolConfig) -> None:
        self.config = config
        self.name = config.name
        self.description = config.description

    def get_parameters_schema(self) -> dict:
        return self.config.parameters or {"type": "object", "properties": {}}

    def run(self, arguments: dict) -> ToolResult:
        started_at = datetime.utcnow()
        status_code: int | None = None
        try:
            method = self.config.method.upper()
            if method == "POST":
                response = requests.post(
                    self.config.endpoint,
                    json=arguments,
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                )
            elif method == "GET":
                response = requests.get(
                    self.config.endpoint,
                    params=arguments,
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                )
            else:
                return self._build_result(
                    success=False,
                    started_at=started_at,
                    status_code=status_code,
                    error=f"Unsupported method: {self.config.method}",
                )

            status_code = response.status_code
            result = _parse_response(response)
            if not 200 <= response.status_code < 300:
                return self._build_result(
                    success=False,
                    started_at=started_at,
                    status_code=status_code,
                    error=f"Remote tool HTTP error: {response.status_code}",
                    result=result,
                )

            return self._build_result(
                success=True,
                started_at=started_at,
                status_code=status_code,
                result=result,
            )
        except Exception as exc:
            return self._build_result(
                success=False,
                started_at=started_at,
                status_code=status_code,
                error=str(exc),
            )

    def _build_result(
        self,
        success: bool,
        started_at: datetime,
        status_code: int | None,
        result: dict | str | None = None,
        error: str | None = None,
    ) -> ToolResult:
        finished_at = datetime.utcnow()
        duration_ms = round((finished_at - started_at).total_seconds() * 1000, 2)
        return ToolResult(
            name=self.name,
            success=success,
            result=result,
            error=error,
            metadata={
                "endpoint": self.config.endpoint,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )


def _parse_response(response: requests.Response) -> dict | str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    return response.text
