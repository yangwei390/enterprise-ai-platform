from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from backend.app.mcp.schemas import MCPServerConfig
from backend.app.mcp.transports.base import BaseMCPTransport, build_client_session
from mcp import ClientSession
from mcp.client.sse import sse_client


class SSECompatMCPTransport(BaseMCPTransport):
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    @asynccontextmanager
    async def session_context(self) -> AsyncIterator[ClientSession]:
        async with sse_client(
            str(self.config.url),
            headers=self.config.headers,
            timeout=self.config.connect_timeout_seconds,
            sse_read_timeout=max(self.config.timeout_seconds, 30),
        ) as (read_stream, write_stream):
            async with build_client_session(read_stream, write_stream) as session:
                yield session

    async def close(self) -> None:
        return None
