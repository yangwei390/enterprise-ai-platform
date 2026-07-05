from abc import ABC, abstractmethod

from backend.app.llms.base import LLMRequest, LLMResponse


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
