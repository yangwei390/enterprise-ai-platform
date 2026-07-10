import asyncio
import re
from typing import Any

from backend.app.mcp.client_manager import MCPClientManager, get_mcp_client_manager
from backend.app.mcp.schemas import MCPServerConfig
from backend.app.mcp.security import enforce_tool_policy
from backend.app.tools.base import BaseTool, ToolDescriptor, ToolResult
from pydantic import BaseModel, ConfigDict


class MCPToolArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class MCPToolAdapter(BaseTool):
    args_schema: type[BaseModel] = MCPToolArgs
    source = "mcp"
    permission = "public"

    def __init__(
        self,
        *,
        server_config: MCPServerConfig,
        remote_tool: Any,
        manager: MCPClientManager | None = None,
    ) -> None:
        self.server_config = server_config
        self.remote_tool = remote_tool
        self.remote_tool_name = str(getattr(remote_tool, "name", ""))
        self.name = build_mcp_tool_name(server_config.name, self.remote_tool_name)
        self.description = str(
            getattr(remote_tool, "description", None)
            or f"MCP tool {self.remote_tool_name} from {server_config.name}"
        )
        self.manager = manager

    def get_parameters_schema(self) -> dict:
        schema: Any = getattr(self.remote_tool, "inputSchema", None)
        if schema is None:
            schema = getattr(self.remote_tool, "input_schema", None)
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        return schema if isinstance(schema, dict) else {"type": "object", "properties": {}}

    def get_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            description=self.description,
            input_schema=self.get_parameters_schema(),
            output_schema=None,
            provider="mcp",
            version=None,
            enabled=True,
            async_supported=True,
            tags=["mcp", *self.server_config.tags],
            metadata={
                "permission": self.permission,
                "required_permissions": self.server_config.required_permissions,
                "mcp_server": self.server_config.name,
                "mcp_remote_tool_name": self.remote_tool_name,
                "mcp_transport": self.server_config.transport,
                "mcp_protocol_version": None,
                "mcp_server_info": self.server_config.metadata.get("server_info"),
                "discovery_source": "mcp",
                "dynamic_registration": True,
            },
        )

    def run(self, arguments: dict) -> ToolResult:
        return asyncio.run(self.arun(arguments))

    async def arun(self, arguments: dict) -> ToolResult:
        try:
            enforce_tool_policy(
                required_permissions=self.server_config.required_permissions,
                granted_permissions=arguments.pop("_permissions", None),
            )
            manager = self.manager or get_mcp_client_manager()
            result = await manager.call_tool(
                self.server_config.name,
                self.remote_tool_name,
                arguments,
            )
            return _to_tool_result(
                registered_name=self.name,
                server_name=self.server_config.name,
                remote_tool_name=self.remote_tool_name,
                transport=self.server_config.transport,
                result=result,
            )
        except Exception as exc:
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                metadata={
                    "provider": "mcp",
                    "mcp_server": self.server_config.name,
                    "mcp_remote_tool_name": self.remote_tool_name,
                    "mcp_transport": self.server_config.transport,
                    "error_code": type(exc).__name__,
                },
            )


def build_mcp_tool_name(server_name: str, remote_tool_name: str) -> str:
    return f"mcp__{_sanitize_name(server_name)}__{_sanitize_name(remote_tool_name)}"


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not sanitized or not re.match(r"[a-zA-Z_]", sanitized[0]):
        sanitized = f"mcp_{sanitized}"
    return sanitized[:48]


def _to_tool_result(
    *,
    registered_name: str,
    server_name: str,
    remote_tool_name: str,
    transport: str,
    result: Any,
) -> ToolResult:
    dumped = _dump_result(result)
    is_error = bool(dumped.get("isError") or dumped.get("is_error"))
    content = dumped.get("content") or []
    structured_content = dumped.get("structuredContent") or dumped.get("structured_content")
    text = _extract_text(content)
    payload = {
        "content": content,
        "text": text,
        "structured_content": structured_content,
        "is_error": is_error,
        "metadata": {
            "mcp_server": server_name,
            "mcp_remote_tool_name": remote_tool_name,
        },
    }
    return ToolResult(
        name=registered_name,
        success=not is_error,
        result=payload,
        error=text if is_error else None,
        metadata={
            "provider": "mcp",
            "mcp_server": server_name,
            "mcp_remote_tool_name": remote_tool_name,
            "mcp_transport": transport,
            "is_error": is_error,
        },
    )


def _dump_result(result: Any) -> dict:
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(result, dict):
        return result
    return {}


def _extract_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for item in content:
        if hasattr(item, "model_dump"):
            item = item.model_dump(by_alias=True)
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)
