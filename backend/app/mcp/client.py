import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

from backend.app.logger import logger
from backend.app.mcp.errors import MCPConnectionError
from backend.app.mcp.schemas import MCPHealthStatus, MCPServerConfig
from backend.app.mcp.security import redact_mapping
from backend.app.mcp.transports import (
    BaseMCPTransport,
    SSECompatMCPTransport,
    StdioMCPTransport,
    StreamableHTTPMCPTransport,
)
from mcp import ClientSession


class MCPClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.state = "disconnected"
        self.session: ClientSession | None = None
        self.initialize_result: Any | None = None
        self.connected_at: datetime | None = None
        self.last_error: str | None = None
        self._stack: AsyncExitStack | None = None
        self._connect_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()
        self._transport = _build_transport(config)

    async def connect(self) -> None:
        if self.session is not None and self.state == "connected":
            return
        async with self._connect_lock:
            if self.session is not None and self.state == "connected":
                return
            await self.disconnect()
            self.state = "connecting"
            started_at = perf_counter()
            try:
                stack = AsyncExitStack()
                session = await stack.enter_async_context(self._transport.session_context())
                async with asyncio.timeout(self.config.connect_timeout_seconds):
                    self.initialize_result = await session.initialize()
                self._stack = stack
                self.session = session
                self.connected_at = datetime.now(UTC)
                self.last_error = None
                self.state = "connected"
                logger.info(
                    "MCP client connected | "
                    f"server={self.config.name} | transport={self.config.transport} | "
                    f"duration_ms={round((perf_counter() - started_at) * 1000, 2)}"
                )
            except Exception as exc:
                self.state = "failed"
                self.last_error = str(exc)
                logger.exception(f"MCP client connect failed | server={self.config.name}")
                await self.disconnect()
                raise MCPConnectionError(str(exc)) from exc

    async def disconnect(self) -> None:
        if self._stack is None and self.session is None:
            self.state = "disconnected"
            return
        self.state = "closing"
        stack = self._stack
        self._stack = None
        self.session = None
        try:
            if stack is not None:
                await stack.aclose()
            await self._transport.close()
        except asyncio.CancelledError:
            logger.debug(
                f"MCP client disconnect received internal cancellation | "
                f"server={self.config.name}"
            )
        finally:
            self.state = "disconnected"

    async def list_tools(self) -> Any:
        await self._ensure_connected()
        if self.session is None:
            raise MCPConnectionError("MCP session is not connected")
        async with asyncio.timeout(self.config.timeout_seconds):
            return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: dict) -> Any:
        await self._ensure_connected()
        if self.session is None:
            raise MCPConnectionError("MCP session is not connected")
        async with self._call_lock:
            async with asyncio.timeout(self.config.tool_call_timeout_seconds):
                return await self.session.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=timedelta(
                        seconds=self.config.tool_call_timeout_seconds
                    ),
                )

    async def health(self) -> MCPHealthStatus:
        return MCPHealthStatus(
            server_name=self.config.name,
            transport=self.config.transport,
            connected=self.session is not None and self.state == "connected",
            state=self.state,
            healthy=self.session is not None and self.state == "connected",
            last_error=self.last_error,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict:
        initialize_dump = _model_dump(self.initialize_result)
        return {
            "server_name": self.config.name,
            "transport": self.config.transport,
            "connected": self.session is not None and self.state == "connected",
            "protocol_version": initialize_dump.get("protocolVersion"),
            "server_info": initialize_dump.get("serverInfo"),
            "capabilities": initialize_dump.get("capabilities"),
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_error": self.last_error,
            "headers": redact_mapping(self.config.headers),
        }

    async def _ensure_connected(self) -> None:
        if self.session is not None and self.state == "connected":
            return
        if not self.config.auto_reconnect:
            raise MCPConnectionError("MCP client is not connected")
        await self.connect()


def _build_transport(config: MCPServerConfig) -> BaseMCPTransport:
    if config.transport == "stdio":
        return StdioMCPTransport(config)
    if config.transport == "streamable_http":
        return StreamableHTTPMCPTransport(config)
    if config.transport == "sse":
        return SSECompatMCPTransport(config)
    raise ValueError(f"unsupported MCP transport: {config.transport}")


def _model_dump(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}
