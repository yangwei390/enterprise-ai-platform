from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp import ClientSession


class BaseMCPTransport(ABC):
    @abstractmethod
    def session_context(self) -> AbstractAsyncContextManager[Any]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    async def health(self) -> dict:
        return {"healthy": True}


def build_client_session(read_stream: Any, write_stream: Any) -> ClientSession:
    return ClientSession(read_stream, write_stream)
