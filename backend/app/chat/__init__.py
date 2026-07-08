from typing import TYPE_CHECKING

from backend.app.chat.base import ChatRequest, ChatResponse, ChatSource

if TYPE_CHECKING:
    from backend.app.chat.service import ChatService

__all__ = ["ChatRequest", "ChatResponse", "ChatService", "ChatSource"]


def __getattr__(name: str):
    if name == "ChatService":
        from backend.app.chat.service import ChatService

        return ChatService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
