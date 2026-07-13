from abc import ABC, abstractmethod
from collections.abc import Iterator

from backend.app.llms.base import LLMRequest, LLMResponse


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> Iterator[str]:
        response = self.chat(request)
        yield response.answer
