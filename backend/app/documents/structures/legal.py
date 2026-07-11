import re

from backend.app.config.settings import settings
from backend.app.documents.schemas import DocumentStructure, DocumentStructureNode
from backend.app.documents.structure import CHINESE_NUMBER_PATTERN, parse_chinese_number
from backend.app.documents.structures.base import BaseStructureParser
from backend.app.documents.structures.plain_text import PlainTextStructureParser


class LegalStructureParser(BaseStructureParser):
    document_type = "legal"
    chapter_pattern = re.compile(
        rf"(?m)^\s*(?P<label>第(?P<number>{CHINESE_NUMBER_PATTERN})(?P<kind>编|章))\s*(?P<title>[^\n]*)"
    )
    section_pattern = re.compile(
        rf"(?m)^\s*(?P<label>第(?P<number>{CHINESE_NUMBER_PATTERN})节)\s*(?P<title>[^\n]*)"
    )
    article_pattern = re.compile(
        rf"(?m)^\s*(?P<label>第(?P<number>{CHINESE_NUMBER_PATTERN})条)\s*"
    )
    law_title_pattern = re.compile(
        r"(?m)^\s*(?P<title>中华人民共和国[^\n]{0,30}(法|条例|办法|规定))\s*$"
    )

    def parse(self, text: str, metadata: dict) -> DocumentStructure:
        try:
            return self._parse(text=text, metadata=metadata)
        except Exception:
            structure = PlainTextStructureParser().parse(text, metadata)
            return structure.model_copy(
                update={
                    "document_type": "plain_text",
                    "metadata": {
                        **structure.metadata,
                        "parse_failed": True,
                        "fallback_from": "legal",
                    },
                }
            )

    def _parse(self, text: str, metadata: dict) -> DocumentStructure:
        title_match = self.law_title_pattern.search(text)
        title = (
            title_match.group("title").strip()
            if title_match
            else str(metadata.get("source") or "legal_document")
        )
        root = DocumentStructureNode(
            id="root",
            node_type="document",
            title=title,
            text=text,
            level=0,
            order=0,
            path=[title],
            start_offset=0,
            end_offset=len(text),
            metadata={"document_type": "legal"},
        )
        nodes = [root]

        chapter_matches = list(self.chapter_pattern.finditer(text))
        section_matches = list(self.section_pattern.finditer(text))
        article_matches = list(self.article_pattern.finditer(text))
        if not chapter_matches and not article_matches:
            return PlainTextStructureParser().parse(text, metadata)

        chapter_nodes = self._build_chapters(text, root, chapter_matches)
        nodes.extend(chapter_nodes)
        nodes_by_id = {node.id: node for node in nodes}
        section_nodes = self._build_sections(text, root, chapter_nodes, section_matches)
        nodes.extend(section_nodes)
        nodes_by_id.update({node.id: node for node in section_nodes})
        article_nodes = self._build_articles(
            text=text,
            root=root,
            chapter_nodes=chapter_nodes,
            section_nodes=section_nodes,
            article_matches=article_matches,
        )
        nodes.extend(article_nodes)

        for node in nodes[1:]:
            if node.parent_id and node.parent_id in nodes_by_id:
                parent = nodes_by_id[node.parent_id]
                if node.id not in parent.children:
                    parent.children.append(node.id)

        max_depth = max((node.level for node in nodes), default=0)
        return DocumentStructure(
            document_id=metadata.get("document_id"),
            document_type="legal",
            root_id=root.id,
            nodes=nodes[: settings.DOCUMENT_STRUCTURE_MAX_NODES],
            metadata={
                "node_count": len(nodes),
                "max_depth": max_depth,
                "parse_failed": False,
                "chapter_count": len(chapter_nodes),
                "article_count": len(article_nodes),
            },
        )

    def _build_chapters(
        self,
        text: str,
        root: DocumentStructureNode,
        matches: list[re.Match],
    ) -> list[DocumentStructureNode]:
        chapters = []
        for index, match in enumerate(matches):
            end_offset = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            label = match.group("label")
            title = match.group("title").strip()
            chapter_title = _clean_title(title)
            full_title = f"{label} {chapter_title}".strip()
            path = [*root.path, full_title]
            chapters.append(
                DocumentStructureNode(
                    id=f"chapter_{index + 1}",
                    node_type="chapter",
                    title=full_title,
                    text=text[match.start():end_offset].strip(),
                    level=1,
                    order=index + 1,
                    parent_id=root.id,
                    path=path,
                    start_offset=match.start(),
                    end_offset=end_offset,
                    metadata={
                        "chapter_number": parse_chinese_number(match.group("number")),
                        "chapter_label": label,
                        "chapter_title": chapter_title,
                        "section_path": path,
                    },
                )
            )
        return chapters

    def _build_sections(
        self,
        text: str,
        root: DocumentStructureNode,
        chapter_nodes: list[DocumentStructureNode],
        matches: list[re.Match],
    ) -> list[DocumentStructureNode]:
        sections = []
        for index, match in enumerate(matches):
            parent = _find_parent_node(match.start(), chapter_nodes) or root
            end_offset = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else parent.end_offset
            )
            label = match.group("label")
            title = _clean_title(match.group("title"))
            full_title = f"{label} {title}".strip()
            path = [*parent.path, full_title]
            sections.append(
                DocumentStructureNode(
                    id=f"section_{index + 1}",
                    node_type="section",
                    title=full_title,
                    text=text[match.start():end_offset].strip(),
                    level=2,
                    order=index + 1,
                    parent_id=parent.id,
                    path=path,
                    start_offset=match.start(),
                    end_offset=end_offset,
                    metadata={
                        **parent.metadata,
                        "section_number": parse_chinese_number(match.group("number")),
                        "section_label": label,
                        "section_title": title,
                        "section_path": path,
                    },
                )
            )
        return sections

    def _build_articles(
        self,
        *,
        text: str,
        root: DocumentStructureNode,
        chapter_nodes: list[DocumentStructureNode],
        section_nodes: list[DocumentStructureNode],
        article_matches: list[re.Match],
    ) -> list[DocumentStructureNode]:
        articles = []
        parents = [*section_nodes, *chapter_nodes]
        for index, match in enumerate(article_matches):
            parent = _find_parent_node(match.start(), parents) or root
            next_article_offset = (
                article_matches[index + 1].start()
                if index + 1 < len(article_matches)
                else parent.end_offset
            )
            parent_end = parent.end_offset or len(text)
            end_offset = min(next_article_offset or parent_end, parent_end)
            label = match.group("label")
            number = parse_chinese_number(match.group("number"))
            path = [*parent.path, label]
            articles.append(
                DocumentStructureNode(
                    id=f"article_{number or index + 1}",
                    node_type="article",
                    title=label,
                    text=text[match.start():end_offset].strip(),
                    level=parent.level + 1,
                    order=index + 1,
                    parent_id=parent.id,
                    path=path,
                    start_offset=match.start(),
                    end_offset=end_offset,
                    metadata={
                        **parent.metadata,
                        "article_number": number,
                        "article_label": label,
                        "section_path": parent.path,
                    },
                )
            )
        return articles


def _find_parent_node(
    offset: int,
    nodes: list[DocumentStructureNode],
) -> DocumentStructureNode | None:
    candidates = [
        node
        for node in nodes
        if node.start_offset is not None
        and node.end_offset is not None
        and node.start_offset <= offset < node.end_offset
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.level)


def _clean_title(title: str | None) -> str:
    return " ".join(str(title or "").strip().split())
