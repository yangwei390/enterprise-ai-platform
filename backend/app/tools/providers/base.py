from abc import ABC, abstractmethod

from backend.app.tools.base import BaseTool


class BaseToolProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def discover(self) -> list[BaseTool]:
        raise NotImplementedError

    async def adiscover(self) -> list[BaseTool]:
        return self.discover()

    def health(self) -> dict:
        return {"provider": self.name, "healthy": True}
