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
        assistants = [
            AgentAssistant(
                id="general_agent",
                name="通用 AI 助手",
                description="理解用户任务，并根据当前系统能力完成问答、计算和资料整理。",
                capabilities=self._general_capabilities(enabled_tools),
                recommended="knowledge_search" not in enabled_tools,
                metadata=self._public_metadata(registry.version),
            )
        ]
        if "knowledge_search" in enabled_tools:
            assistants.insert(
                0,
                AgentAssistant(
                    id="knowledge_agent",
                    name="知识库问答助手",
                    description="面向企业知识库问答，适合查询制度、文档条款和业务资料。",
                    capabilities=[
                        "查询企业知识库",
                        "基于资料整理答案",
                        "在可用时返回来源引用",
                        "支持围绕同一会话继续追问",
                    ],
                    recommended=True,
                    metadata=self._public_metadata(registry.version),
                ),
            )
        return assistants

    def get_assistant(self, agent_id: str) -> AgentAssistant | None:
        return next((agent for agent in self.list_assistants() if agent.id == agent_id), None)

    def _general_capabilities(self, enabled_tools: set[str]) -> list[str]:
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

    def _public_metadata(self, registry_version: int) -> dict:
        return {
            "runtime": "langgraph_v2",
            "async_enabled": settings.AGENT_ASYNC_ENABLED,
            "registry_version": registry_version,
        }
