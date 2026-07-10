from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    source: str = "builtin"
    permission: str = "public"


class ToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict | None = None
    provider: str = "builtin"
    version: str | None = None
    enabled: bool = True
    async_supported: bool = False
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
            source=self.provider,
            permission=str(self.metadata.get("permission", "public")),
        )

    def to_llm_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


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
    name: str
    description: str
    args_schema: type[BaseModel]
    source: str = "builtin"
    permission: str = "public"

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
            source=self.source,
            permission=self.permission,
        )

    def get_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            description=self.description,
            input_schema=self.get_parameters_schema(),
            provider=self.source,
            enabled=True,
            async_supported=callable(getattr(self, "arun", None)),
            metadata={
                "permission": self.permission,
                "required_permissions": [],
            },
        )
