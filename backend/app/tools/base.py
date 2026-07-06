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
    metadata: dict = Field(default_factory=dict)


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, arguments: dict) -> ToolResult:
        raise NotImplementedError

    def get_parameters_schema(self) -> dict:
        return self.args_schema.model_json_schema()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters_schema(),
        )
