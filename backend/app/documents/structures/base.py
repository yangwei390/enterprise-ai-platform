from abc import ABC, abstractmethod

from backend.app.documents.schemas import DocumentStructure


class BaseStructureParser(ABC):
    document_type: str

    @abstractmethod
    def parse(self, text: str, metadata: dict) -> DocumentStructure:
        raise NotImplementedError
