from backend.app.llms.base import BaseLLM, LLMRequest, LLMResponse
from backend.app.llms.clients import DummyLLMClient


class DummyLLMProvider(BaseLLM):
    def __init__(self, client: DummyLLMClient | None = None) -> None:
        self.client = client or DummyLLMClient()

    def chat(self, request: LLMRequest) -> LLMResponse:
        response = self.client.chat(request)
        response.metadata["provider"] = "dummy"
        return response
