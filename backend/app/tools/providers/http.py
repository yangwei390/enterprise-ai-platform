import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import httpx
from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.logger import logger
from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.providers.base import BaseToolProvider
from pydantic import BaseModel, ConfigDict


class HTTPToolArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class HTTPTool(BaseTool):
    args_schema: type[BaseModel] = HTTPToolArgs
    source: str = "http"
    permission: str = "public"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.name = str(config["name"])
        self.description = str(config.get("description") or self.name)
        self.timeout_seconds = int(
            config.get("timeout_seconds") or settings.HTTP_TOOL_DEFAULT_TIMEOUT_SECONDS
        )

    def get_parameters_schema(self) -> dict:
        schema = self.config.get("input_schema")
        return schema if isinstance(schema, dict) else {"type": "object", "properties": {}}

    def run(self, arguments: dict) -> ToolResult:
        return asyncio.run(self.arun(arguments))

    async def arun(self, arguments: dict) -> ToolResult:
        started_at = perf_counter()
        method = str(self.config.get("method", "POST")).upper()
        url = str(self.config["url"])
        headers = _resolve_headers(self.config.get("headers", {}))
        status_code: int | None = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                if method == "POST":
                    response = await client.post(url, json=arguments, headers=headers)
                elif method == "GET":
                    response = await client.get(url, params=arguments, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP tool method: {method}")
            status_code = response.status_code
            result = _parse_httpx_response(response)
            return ToolResult(
                name=self.name,
                success=200 <= response.status_code < 300,
                result=result,
                error=None
                if 200 <= response.status_code < 300
                else f"HTTP tool error: {response.status_code}",
                metadata={
                    "provider": "http",
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
        except Exception as exc:
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                metadata={
                    "provider": "http",
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )


class HTTPToolProvider(BaseToolProvider):
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path or settings.HTTP_TOOL_CONFIG_PATH)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        self.errors: list[str] = []

    @property
    def name(self) -> str:
        return "http"

    def discover(self) -> list[BaseTool]:
        self.errors = []
        if not settings.HTTP_TOOL_PROVIDER_ENABLED or not self.config_path.exists():
            return []

        try:
            parsed = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.errors.append(str(exc))
            logger.exception("HTTP tool config loading failed")
            return []

        raw_tools = parsed.get("tools", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(raw_tools, list):
            return []

        tools: list[BaseTool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict) or not raw_tool.get("enabled", True):
                continue
            try:
                self._validate_config(raw_tool)
                tools.append(HTTPTool(raw_tool))
            except Exception as exc:
                self.errors.append(f"{raw_tool.get('name')}: {exc}")
        return tools

    def health(self) -> dict:
        return {
            "provider": self.name,
            "healthy": not self.errors,
            "errors": self.errors,
            "config_path": str(self.config_path),
        }

    def _validate_config(self, config: dict) -> None:
        for required_key in ["name", "description", "url"]:
            if not config.get(required_key):
                raise ValueError(f"HTTP tool missing {required_key}")
        parsed = urlparse(str(config["url"]))
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("HTTP tool URL must use http or https")


def _resolve_headers(headers: object) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        text = str(value)
        if text.startswith("${") and text.endswith("}"):
            text = os.getenv(text[2:-1], "")
        resolved[str(key)] = text
    return resolved


def _parse_httpx_response(response: httpx.Response) -> dict | str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    return response.text
