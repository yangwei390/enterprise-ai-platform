from pydantic import BaseModel, Field


class MCPToolConfig(BaseModel):
    name: str
    description: str
    endpoint: str
    method: str = "POST"
    headers: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    timeout: int = 30


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class MCPToolCallResult(BaseModel):
    name: str
    success: bool
    result: dict | str | None = None
    error: str | None = None
    metadata: dict = Field(default_factory=dict)
