from backend.app.documents.schemas import DocumentStructure, DocumentStructureNode
from backend.app.documents.structures.base import BaseStructureParser


class PlainTextStructureParser(BaseStructureParser):
    document_type = "plain_text"

    def parse(self, text: str, metadata: dict) -> DocumentStructure:
        root = DocumentStructureNode(
            id="root",
            node_type="document",
            title=str(metadata.get("source") or "document"),
            text=text,
            level=0,
            order=0,
            path=[str(metadata.get("source") or "document")],
            start_offset=0,
            end_offset=len(text),
            metadata={"document_type": "plain_text"},
        )
        return DocumentStructure(
            document_id=metadata.get("document_id"),
            document_type="plain_text",
            root_id=root.id,
            nodes=[root],
            metadata={
                "node_count": 1,
                "max_depth": 0,
                "parse_failed": False,
            },
        )
