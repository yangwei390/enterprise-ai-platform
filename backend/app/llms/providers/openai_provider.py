from backend.app.llms.base import BaseLLM, LLMRequest, LLMResponse
from backend.app.llms.clients.openai_client import OpenAIClient
from backend.app.llms.config import LLMConfig


class OpenAIProvider(BaseLLM):
    supports_tool_calling = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = OpenAIClient(config=config)

    def chat(self, request: LLMRequest) -> LLMResponse:
        response = self.client.chat(request)
        response.metadata["provider"] = "openai"
        return response
