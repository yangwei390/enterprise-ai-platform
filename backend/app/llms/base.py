from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.2
    metadata: dict = Field(default_factory=dict)


class LLMToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    answer: str
    model: str
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
