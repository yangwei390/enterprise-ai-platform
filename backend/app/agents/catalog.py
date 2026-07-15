from backend.app.agents.definition import AgentDefinition, get_agent_definition_registry
from backend.app.agents.schemas import AgentAssistant
from backend.app.config.settings import settings
from backend.app.tools import get_tool_registry


class AgentCatalog:
    def list_assistants(self) -> list[AgentAssistant]:
        registry = get_tool_registry()
        enabled_tools = {
            descriptor.name
            for descriptor in registry.list_descriptors(enabled_only=True)
        }
        definitions = get_agent_definition_registry().list()
        return [
            AgentAssistant(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                capabilities=self._capabilities(definition, enabled_tools),
                recommended=definition.id == self._recommended_agent_id(enabled_tools),
                metadata=self._public_metadata(definition, registry.version),
            )
            for definition in definitions
        ]

    def get_assistant(self, agent_id: str) -> AgentAssistant | None:
        return next((agent for agent in self.list_assistants() if agent.id == agent_id), None)

    def _capabilities(
        self,
        definition: AgentDefinition,
        enabled_tools: set[str],
    ) -> list[str]:
        if definition.id == "knowledge_agent":
            return [
                "查询企业知识库",
                "基于资料整理答案",
                "在可用时返回来源引用",
                "支持围绕同一会话继续追问",
            ]
        capabilities = [
            "分析用户问题",
            "基于上下文整理答案",
            "支持围绕同一会话继续追问",
        ]
        if {"calculator", "current_time"} & enabled_tools:
            capabilities.append("执行轻量任务辅助回答")
        if "knowledge_search" in enabled_tools:
            capabilities.append("在需要时查询企业知识库")
        if any(tool_name.startswith("mcp__") for tool_name in enabled_tools):
            capabilities.append("使用已接入的业务能力完成任务")
        return capabilities

    def _recommended_agent_id(self, enabled_tools: set[str]) -> str:
        return "knowledge_agent" if "knowledge_search" in enabled_tools else "general_agent"

    def _public_metadata(self, definition: AgentDefinition, registry_version: int) -> dict:
        return {
            "runtime": "langgraph_v2",
            "async_enabled": settings.AGENT_ASYNC_ENABLED,
            "definition_version": definition.version,
            "planner_strategy": definition.planner_strategy,
            "default_knowledge_base_id": definition.default_knowledge_base_id,
            "output_mode": definition.output_mode,
            "registry_version": registry_version,
        }
