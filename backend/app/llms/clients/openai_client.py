from typing import Any

from backend.app.exceptions import BusinessException
from backend.app.llms.base import LLMRequest, LLMResponse
from backend.app.llms.clients.base_client import BaseLLMClient
from backend.app.llms.config import LLMConfig
from backend.app.logger import logger
from openai import OpenAI  # type: ignore[reportMissingImports]


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)

    def chat(self, request: LLMRequest) -> LLMResponse:
        try:
            create_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                "temperature": request.temperature,
            }
            if self.config.max_tokens is not None:
                create_kwargs["max_output_tokens"] = self.config.max_tokens

            response = self.client.responses.create(**create_kwargs)
            return LLMResponse(
                answer=response.output_text or "",
                model=self.config.model,
                usage={},
                metadata={
                    "provider": "openai",
                    "client": "openai",
                    "response_id": response.id,
                },
            )
        except Exception as exc:
            logger.exception("OpenAI model call failed")
            raise BusinessException(50010, "OpenAI模型调用失败") from exc
