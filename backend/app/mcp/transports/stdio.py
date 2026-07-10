from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from backend.app.config.settings import PROJECT_ROOT
from backend.app.mcp.schemas import MCPServerConfig
from backend.app.mcp.transports.base import BaseMCPTransport, build_client_session
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class StdioMCPTransport(BaseMCPTransport):
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    @asynccontextmanager
    async def session_context(self) -> AsyncIterator[ClientSession]:
        cwd = self.config.cwd
        if cwd and not Path(cwd).is_absolute():
            cwd = str(PROJECT_ROOT / cwd)
        params = StdioServerParameters(
            command=str(self.config.command),
            args=self.config.args,
            env=self.config.env or None,
            cwd=cwd,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with build_client_session(read_stream, write_stream) as session:
                yield session

    async def close(self) -> None:
        return None
