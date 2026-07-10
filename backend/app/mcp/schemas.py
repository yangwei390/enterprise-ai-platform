from typing import Literal

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    name: str
    enabled: bool = True
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 30
    connect_timeout_seconds: float = 10
    tool_call_timeout_seconds: float = 30
    retry_count: int = 1
    auto_reconnect: bool = True
    required_permissions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class MCPHealthStatus(BaseModel):
    server_name: str
    transport: str
    connected: bool = False
    state: str = "disconnected"
    healthy: bool = False
    last_error: str | None = None
    metadata: dict = Field(default_factory=dict)


class MCPDiscoveryResult(BaseModel):
    connected_servers: list[str] = Field(default_factory=list)
    failed_servers: list[dict] = Field(default_factory=list)
    discovered_tools: int = 0
    duration_ms: float = 0
    errors: list[dict] = Field(default_factory=list)
