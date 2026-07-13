from collections.abc import Iterator

from backend.app.llms.base import LLMRequest, LLMResponse
from backend.app.llms.clients.base_client import BaseLLMClient
from backend.app.llms.config import LLMConfig


class DummyLLMClient(BaseLLMClient):
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.model = self.config.model

    def chat(self, request: LLMRequest) -> LLMResponse:
        user_message = self._get_last_user_message(request)
        return LLMResponse(
            answer=(
                "这是 DummyLLM 的模拟回答。"
                "当前系统已经完成 Prompt 构建，后续会替换为真实大模型。"
            ),
            model=self.model,
            usage={},
            metadata={
                "client": "dummy",
                "provider": self.config.provider,
                "model": self.model,
                "temperature": self.config.temperature,
                "message_count": len(request.messages),
                "has_context": "上下文：" in user_message,
            },
        )

    def stream(self, request: LLMRequest) -> Iterator[str]:
        response = self.chat(request)
        for index in range(0, len(response.answer), 8):
            yield response.answer[index : index + 8]

    def _get_last_user_message(self, request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        return ""
