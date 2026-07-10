import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.mcp.errors import MCPConfigError, MCPPermissionError
from backend.app.mcp.schemas import MCPServerConfig

SENSITIVE_KEYWORDS = ("authorization", "cookie", "token", "secret", "password", "key")


def resolve_env_value(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, "")
    return value


def resolve_env_mapping(values: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in values.items():
        resolved[str(key)] = resolve_env_value(str(value))
    return resolved


def redact_mapping(values: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in values.items():
        if _is_sensitive_key(key):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def hash_arguments(arguments: dict) -> str:
    return hashlib.sha256(repr(sorted(arguments.items())).encode("utf-8")).hexdigest()


def validate_server_config(config: MCPServerConfig) -> None:
    if config.transport == "stdio":
        validate_stdio_config(config)
    elif config.transport == "streamable_http":
        validate_streamable_http_config(config)
    elif config.transport == "sse":
        validate_sse_config(config)
    else:
        raise MCPConfigError(f"unsupported MCP transport: {config.transport}")


def validate_stdio_config(config: MCPServerConfig) -> None:
    if not settings.MCP_STDIO_ENABLED:
        raise MCPPermissionError("stdio MCP transport is disabled")
    if not config.command:
        raise MCPConfigError("stdio MCP server requires command")
    allowed = {
        item.strip()
        for item in settings.MCP_STDIO_ALLOWED_COMMANDS.split(",")
        if item.strip()
    }
    command_name = Path(config.command).name
    if command_name not in allowed:
        raise MCPPermissionError(f"stdio command is not allowed: {command_name}")
    if config.cwd:
        cwd = Path(config.cwd)
        if not cwd.is_absolute():
            cwd = PROJECT_ROOT / cwd
        try:
            cwd.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise MCPPermissionError("stdio cwd must stay inside project root") from exc


def validate_streamable_http_config(config: MCPServerConfig) -> None:
    if not settings.MCP_STREAMABLE_HTTP_ENABLED:
        raise MCPPermissionError("streamable HTTP MCP transport is disabled")
    _validate_http_url(config.url)


def validate_sse_config(config: MCPServerConfig) -> None:
    if not settings.MCP_SSE_COMPAT_ENABLED:
        raise MCPPermissionError("SSE MCP transport is disabled")
    _validate_http_url(config.url)


def enforce_tool_policy(
    *,
    required_permissions: list[str],
    granted_permissions: list[str] | None = None,
) -> None:
    if not settings.MCP_PERMISSION_ENFORCEMENT_ENABLED:
        return
    if not required_permissions:
        return
    granted = set(granted_permissions or [])
    missing = [permission for permission in required_permissions if permission not in granted]
    if missing:
        raise MCPPermissionError(f"missing MCP permissions: {', '.join(missing)}")


def _validate_http_url(url: str | None) -> None:
    if not url:
        raise MCPConfigError("HTTP MCP server requires url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise MCPConfigError("MCP HTTP url must use http or https")
    if parsed.scheme == "http" and not _is_localhost(parsed.hostname or ""):
        if not settings.MCP_ALLOW_INSECURE_HTTP_LOCALHOST:
            raise MCPPermissionError("insecure HTTP is disabled")
        raise MCPPermissionError("non-localhost MCP HTTP server must use https")


def _is_localhost(hostname: str) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)
