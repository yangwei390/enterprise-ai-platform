from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class PromptBuildRequest(BaseModel):
    query: str
    context_text: str
    system_prompt: str | None = None


class PromptMessage(BaseModel):
    role: str
    content: str


class PromptBuildResult(BaseModel):
    messages: list[PromptMessage]
    prompt_text: str
    metadata: dict = Field(default_factory=dict)


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, request: PromptBuildRequest) -> PromptBuildResult:
        raise NotImplementedError
