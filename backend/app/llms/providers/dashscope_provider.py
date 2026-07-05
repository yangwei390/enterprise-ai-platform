from backend.app.llms.base import BaseLLM, LLMRequest, LLMResponse
from backend.app.llms.clients.dashscope_client import DashScopeClient
from backend.app.llms.config import LLMConfig


class DashScopeProvider(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = DashScopeClient(config=config)

    def chat(self, request: LLMRequest) -> LLMResponse:
        response = self.client.chat(request)
        response.metadata["provider"] = "dashscope"
        return response
