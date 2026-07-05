from backend.app.prompts.base import (
    BasePromptBuilder,
    PromptBuildRequest,
    PromptBuildResult,
    PromptMessage,
)

DEFAULT_SYSTEM_PROMPT = """你是一个企业级知识库问答助手。
请只基于给定上下文回答问题。
如果上下文中没有答案，请回答：根据当前知识库内容无法回答该问题。
回答时尽量简洁、准确。"""


class BasicPromptBuilder(BasePromptBuilder):
    def build(self, request: PromptBuildRequest) -> PromptBuildResult:
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        user_prompt = (
            f"上下文：\n{request.context_text}\n\n"
            f"问题：\n{request.query}\n\n"
            "请基于上下文回答。"
        )
        messages = [
            PromptMessage(role="system", content=system_prompt),
            PromptMessage(role="user", content=user_prompt),
        ]
        prompt_text = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

        return PromptBuildResult(
            messages=messages,
            prompt_text=prompt_text,
            metadata={
                "context_chars": len(request.context_text),
                "query_chars": len(request.query),
                "message_count": len(messages),
            },
        )
