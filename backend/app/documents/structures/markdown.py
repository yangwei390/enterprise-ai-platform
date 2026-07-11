import re

from backend.app.config.settings import settings
from backend.app.documents.schemas import DocumentStructure, DocumentStructureNode
from backend.app.documents.structures.base import BaseStructureParser


class MarkdownStructureParser(BaseStructureParser):
    document_type = "markdown"
    heading_pattern = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")

    def parse(self, text: str, metadata: dict) -> DocumentStructure:
        source = str(metadata.get("source") or "document")
        root = DocumentStructureNode(
            id="root",
            node_type="document",
            title=source,
            text="",
            level=0,
            order=0,
            path=[source],
            start_offset=0,
            end_offset=len(text),
            metadata={"document_type": "markdown"},
        )
        nodes = [root]
        stack: list[DocumentStructureNode] = [root]
        in_fence = False
        offset = 0
        current = root

        for order, line in enumerate(text.splitlines(keepends=True), start=1):
            plain_line = line.rstrip("\n")
            if plain_line.strip().startswith("```"):
                in_fence = not in_fence
                current.text += line
                offset += len(line)
                continue

            match = self.heading_pattern.match(plain_line) if not in_fence else None
            if match:
                level = len(match.group("marks"))
                title = match.group("title").strip()
                while stack and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1] if stack else root
                node_id = f"heading_{len(nodes)}"
                path = [*parent.path, title]
                node = DocumentStructureNode(
                    id=node_id,
                    node_type="heading",
                    title=title,
                    text="",
                    level=level,
                    order=order,
                    parent_id=parent.id,
                    path=path,
                    start_offset=offset,
                    end_offset=offset + len(line),
                    metadata={
                        "heading_level": level,
                        "heading_title": title,
                        "section_path": path,
                    },
                )
                parent.children.append(node.id)
                nodes.append(node)
                stack.append(node)
                current = node
            else:
                current.text += line
                current.end_offset = offset + len(line)
            offset += len(line)
            if len(nodes) >= settings.DOCUMENT_STRUCTURE_MAX_NODES:
                break

        max_depth = max((node.level for node in nodes), default=0)
        return DocumentStructure(
            document_id=metadata.get("document_id"),
            document_type="markdown",
            root_id=root.id,
            nodes=nodes,
            metadata={
                "node_count": len(nodes),
                "max_depth": max_depth,
                "parse_failed": False,
            },
        )
