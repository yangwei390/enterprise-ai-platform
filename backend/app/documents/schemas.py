from pydantic import BaseModel, Field


class DocumentStructureNode(BaseModel):
    id: str
    node_type: str
    title: str | None = None
    text: str = ""
    level: int = 0
    order: int = 0
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)
    start_offset: int | None = None
    end_offset: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentStructure(BaseModel):
    document_id: int | None = None
    document_type: str
    root_id: str
    nodes: list[DocumentStructureNode]
    metadata: dict = Field(default_factory=dict)

    def node_by_id(self) -> dict[str, DocumentStructureNode]:
        return {node.id: node for node in self.nodes}

    def children_of(self, node_id: str) -> list[DocumentStructureNode]:
        nodes = self.node_by_id()
        parent = nodes.get(node_id)
        if parent is None:
            return []
        return [nodes[child_id] for child_id in parent.children if child_id in nodes]
