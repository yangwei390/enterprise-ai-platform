from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    success: bool
    result: dict | str | None = None
    error: str | None = None


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]

    @abstractmethod
    def run(self, arguments: dict) -> ToolResult:
        raise NotImplementedError

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
