import re

from backend.app.memory.base import MemoryMessage


class TokenBudgetManager:
    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_chinese_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
        english_tokens = len([token for token in non_chinese_text.split() if token])
        return chinese_chars + english_tokens

    def trim_messages(
        self,
        messages: list[MemoryMessage],
        max_tokens: int,
    ) -> list[MemoryMessage]:
        if max_tokens <= 0:
            return []

        selected: list[MemoryMessage] = []
        used_tokens = 0
        for message in reversed(messages):
            message_tokens = self.estimate_tokens(message.content)
            if used_tokens + message_tokens > max_tokens:
                continue
            selected.append(message)
            used_tokens += message_tokens

        return list(reversed(selected))

    def estimate_messages_tokens(self, messages: list[MemoryMessage]) -> int:
        return sum(self.estimate_tokens(message.content) for message in messages)
