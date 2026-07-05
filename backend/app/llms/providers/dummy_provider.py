from backend.app.llms.base import BaseLLM, LLMRequest, LLMResponse


class DummyLLMProvider(BaseLLM):
    model = "dummy-llm"

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
                "provider": "dummy",
                "message_count": len(request.messages),
                "has_context": "上下文：" in user_message,
            },
        )

    def _get_last_user_message(self, request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        return ""
