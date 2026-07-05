from backend.app.llms.base import BaseLLM, LLMRequest, LLMResponse
from backend.app.llms.clients import DummyLLMClient
from backend.app.llms.config import LLMConfig


class DummyLLMProvider(BaseLLM):
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.client = DummyLLMClient(config=self.config)

    def chat(self, request: LLMRequest) -> LLMResponse:
        response = self.client.chat(request)
        response.metadata["provider"] = "dummy"
        return response
