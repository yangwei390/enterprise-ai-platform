import json
from typing import Any

from backend.app.exceptions import BusinessException
from backend.app.llms.base import LLMRequest, LLMResponse, LLMToolCall
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
            if request.tools:
                return self._chat_with_tools(request)

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
                raw_response=_to_raw_response(response),
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

    def _chat_with_tools(self, request: LLMRequest) -> LLMResponse:
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_to_chat_payload(message) for message in request.messages],
            "temperature": request.temperature,
            "tools": request.tools,
        }
        if request.tool_choice is not None:
            create_kwargs["tool_choice"] = request.tool_choice
        if self.config.max_tokens is not None:
            create_kwargs["max_tokens"] = self.config.max_tokens

        response = self.client.chat.completions.create(**create_kwargs)
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = getattr(message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        return LLMResponse(
            answer=content,
            model=self.config.model,
            tool_calls=_parse_tool_calls(getattr(message, "tool_calls", None)),
            finish_reason=finish_reason,
            raw_response=_to_raw_response(response),
            usage=_to_raw_response(getattr(response, "usage", None)) or {},
            metadata={
                "provider": "openai",
                "client": "openai",
                "response_id": getattr(response, "id", None),
            },
        )


def _parse_tool_calls(raw_tool_calls: Any) -> list[LLMToolCall]:
    if not raw_tool_calls:
        return []

    tool_calls = []
    for raw_tool_call in raw_tool_calls:
        function = getattr(raw_tool_call, "function", None)
        name = getattr(function, "name", None)
        if not name:
            continue
        arguments = _parse_arguments(getattr(function, "arguments", "{}"))
        tool_calls.append(
            LLMToolCall(
                id=getattr(raw_tool_call, "id", None),
                name=name,
                arguments=arguments,
            )
        )
    return tool_calls


def _message_to_chat_payload(message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


def _parse_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _to_raw_response(response: Any) -> dict | None:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        value = response.model_dump()
        return value if isinstance(value, dict) else None
    if isinstance(response, dict):
        return response
    return None
