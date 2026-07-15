from copy import deepcopy

from backend.app.config.settings import settings
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentDefinitionError(ValueError):
    pass


class AgentDefinitionNotFoundError(AgentDefinitionError):
    pass


class AgentDefinitionDisabledError(AgentDefinitionError):
    pass


class AgentDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: str
    instructions: str
    planner_strategy: str
    tool_allowlist: list[str] = Field(default_factory=list)
    workflow_allowlist: list[str] = Field(default_factory=list)
    default_knowledge_base_id: int | None = None
    memory_policy: dict = Field(default_factory=dict)
    retrieval_policy: dict = Field(default_factory=dict)
    model_settings: dict = Field(default_factory=dict, alias="model_config")
    max_steps: int
    timeout_seconds: int
    output_mode: str
    safety_policy: dict = Field(default_factory=dict)
    enabled: bool = True
    version: str = "1.0"
    metadata: dict = Field(default_factory=dict)

    @field_validator("id", "name", "instructions", "planner_strategy", "output_mode")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("max_steps", "timeout_seconds")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value


class AgentDefinitionRegistry:
    def __init__(
        self,
        definitions: list[AgentDefinition] | None = None,
        *,
        default_agent_id: str = "general_agent",
    ) -> None:
        self.default_agent_id = default_agent_id
        self._definitions: dict[str, AgentDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: AgentDefinition) -> None:
        self._definitions[definition.id] = definition

    def get(self, agent_id: str | None = None, *, allow_disabled: bool = False) -> AgentDefinition:
        selected_agent_id = agent_id or self.default_agent_id
        definition = self._definitions.get(selected_agent_id)
        if definition is None:
            raise AgentDefinitionNotFoundError(
                f"Agent definition not found: {selected_agent_id}"
            )
        if not definition.enabled and not allow_disabled:
            raise AgentDefinitionDisabledError(
                f"Agent definition is disabled: {selected_agent_id}"
            )
        return definition.model_copy(deep=True)

    def list(self, *, enabled_only: bool = True) -> list[AgentDefinition]:
        definitions = list(self._definitions.values())
        if enabled_only:
            definitions = [definition for definition in definitions if definition.enabled]
        return [definition.model_copy(deep=True) for definition in definitions]

    def clear(self) -> None:
        self._definitions.clear()


def _default_model_config(**overrides) -> dict:
    config = {"model": None, "temperature": 0}
    config.update(overrides)
    return config


def _builtin_definitions() -> list[AgentDefinition]:
    return [
        AgentDefinition(
            id="general_agent",
            name="通用 AI 助手",
            description="理解用户任务，并根据当前系统能力完成问答、计算和资料整理。",
            instructions=(
                "你是企业 AI 平台中的通用助手。先理解用户意图，再选择必要工具；"
                "如果不需要工具，直接给出简洁、准确的回答。"
            ),
            planner_strategy=settings.AGENT_PLANNER_STRATEGY,
            tool_allowlist=[
                "calculator",
                "echo",
                "get_current_time",
                "knowledge_search",
                "workflow_default_knowledge",
            ],
            workflow_allowlist=["default_agent_workflow_v2"],
            default_knowledge_base_id=None,
            memory_policy={"enabled": True, "scope": "conversation"},
            retrieval_policy={"enabled": False, "mode": "on_demand"},
            model_config=_default_model_config(temperature=0.2),
            max_steps=settings.AGENT_MAX_STEPS,
            timeout_seconds=settings.AGENT_ASYNC_TIMEOUT_SECONDS,
            output_mode="answer",
            safety_policy={"tool_use_requires_planner": True},
            version="1.0",
            metadata={"capability_group": "general"},
        ),
        AgentDefinition(
            id="knowledge_agent",
            name="知识库问答助手",
            description="面向企业知识库问答，适合查询制度、文档条款和业务资料。",
            instructions=(
                "你是企业知识库问答助手。优先围绕用户问题检索知识库，"
                "回答必须基于可用资料并尽量保留来源信息。"
            ),
            planner_strategy="json_plan",
            tool_allowlist=["knowledge_search"],
            workflow_allowlist=["default_agent_workflow_v2"],
            default_knowledge_base_id=None,
            memory_policy={"enabled": True, "scope": "conversation"},
            retrieval_policy={"enabled": True, "mode": "knowledge_first", "top_k": 8},
            model_config=_default_model_config(temperature=0),
            max_steps=min(settings.AGENT_MAX_STEPS, 8),
            timeout_seconds=settings.AGENT_ASYNC_TIMEOUT_SECONDS,
            output_mode="grounded_answer",
            safety_policy={"require_citations_when_available": True},
            version="1.0",
            metadata={"capability_group": "knowledge"},
        ),
    ]


_registry = AgentDefinitionRegistry(definitions=_builtin_definitions())


def get_agent_definition_registry() -> AgentDefinitionRegistry:
    return _registry


def reset_agent_definition_registry() -> AgentDefinitionRegistry:
    _registry.clear()
    for definition in _builtin_definitions():
        _registry.register(deepcopy(definition))
    return _registry
