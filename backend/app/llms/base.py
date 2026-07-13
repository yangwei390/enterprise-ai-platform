from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.2
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    metadata: dict = Field(default_factory=dict)


class LLMToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    answer: str
    model: str
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    raw_response: dict | None = None
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class BaseLLM(ABC):
    supports_tool_calling: bool = False

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> Iterator[str]:
        response = self.chat(request)
        yield response.answer
