from typing import Generic, TypeVar


RepositoryType = TypeVar("RepositoryType")


class BaseService(Generic[RepositoryType]):
    def __init__(self, repository: RepositoryType) -> None:
        self.repository = repository
