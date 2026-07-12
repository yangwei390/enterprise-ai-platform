from backend.app.retrievers.planning.schemas import ConstraintFieldDefinition


class ConstraintRegistry:
    def __init__(self) -> None:
        self._fields: dict[str, ConstraintFieldDefinition] = {}

    def register(self, definition: ConstraintFieldDefinition) -> None:
        if not definition.field.strip():
            raise ValueError("constraint field is empty")
        if definition.field in self._fields:
            raise ValueError(f"constraint field already registered: {definition.field}")
        self._fields[definition.field] = definition

    def get(self, field: str) -> ConstraintFieldDefinition | None:
        definition = self._fields.get(field)
        if definition is not None:
            return definition
        normalized = field.lower()
        for item in self._fields.values():
            aliases = [alias.lower() for alias in item.aliases]
            if normalized in aliases:
                return item
        return None

    def list_fields(self, enabled_only: bool = True) -> list[ConstraintFieldDefinition]:
        fields = list(self._fields.values())
        if enabled_only:
            return [field for field in fields if field.enabled]
        return fields


def default_constraint_registry() -> ConstraintRegistry:
    registry = ConstraintRegistry()
    for definition in (
        ConstraintFieldDefinition(
            field="document_id",
            value_type="int",
            allowed_operators=["eq", "in"],
            aliases=["文档", "document"],
        ),
        ConstraintFieldDefinition(
            field="knowledge_base_id",
            value_type="int",
            allowed_operators=["eq", "in"],
            aliases=["知识库", "knowledge_base"],
        ),
        ConstraintFieldDefinition(
            field="document_type",
            value_type="str",
            allowed_operators=["eq", "in"],
            aliases=["文档类型", "type"],
        ),
        ConstraintFieldDefinition(
            field="chapter_number",
            value_type="int",
            allowed_operators=["eq", "in"],
            aliases=["章", "chapter"],
        ),
        ConstraintFieldDefinition(
            field="article_number",
            value_type="int",
            allowed_operators=["eq", "in"],
            aliases=["条", "article"],
        ),
        ConstraintFieldDefinition(
            field="heading_title",
            value_type="str",
            allowed_operators=["eq", "contains"],
            aliases=["标题", "heading"],
        ),
        ConstraintFieldDefinition(
            field="section_path",
            value_type="list",
            allowed_operators=["contains"],
            aliases=["路径", "section"],
        ),
    ):
        registry.register(definition)
    return registry
