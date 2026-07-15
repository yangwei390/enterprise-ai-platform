import asyncio
from pathlib import Path

import pytest
from backend.app.agents.definition import (
    AgentDefinition,
    get_agent_definition_registry,
    reset_agent_definition_registry,
)
from backend.app.agents.langgraph.graph import build_agent_graph
from backend.app.agents.langgraph.nodes import FinalNode, PlannerNode, ToolNode
from backend.app.agents.langgraph.planner import LLMPlanner
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.langgraph.tool_calling import AgentDecision, AgentToolCall
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.tools import BaseTool, ToolExecutor, ToolRegistry, ToolResult
from pydantic import BaseModel


class KnowledgeArgs(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    conversation_id: int | None = None
    memory_context: str | None = None


class CalculatorArgs(BaseModel):
    expression: str


class EchoArgs(BaseModel):
    text: str


class FakeKnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Search reusable RAG pipeline"
    args_schema = KnowledgeArgs

    def __init__(self, *, with_evidence: bool = True, evidence_mode: str = "both") -> None:
        self.with_evidence = with_evidence
        self.evidence_mode = evidence_mode
        self.calls = 0

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        if not self.with_evidence:
            return ToolResult(
                name=self.name,
                success=True,
                result={
                    "answer": "",
                    "sources": [],
                    "citations": [],
                    "metadata": {"retriever_mode": "hybrid"},
                },
            )
        sources = [
            {
                "source": "policy-a.pdf",
                "document_id": 101,
                "chunk_index": 1,
            }
        ]
        citations = [
            {
                "source": "policy-a.pdf",
                "document_id": 101,
                "chunk_index": 1,
            }
        ]
        if self.evidence_mode == "sources_only":
            citations = []
        if self.evidence_mode == "citations_only":
            sources = []
        return ToolResult(
            name=self.name,
            success=True,
            result={
                "answer": "基于证据的研究结论。",
                "sources": sources,
                "citations": citations,
                "metadata": {
                    "retriever_mode": "hybrid",
                    "query_understanding": {"intent": "summary"},
                    "document_routing": {"route_type": "DOCUMENT"},
                    "retrieval_strategy": {"strategy": "DOCUMENT"},
                },
            },
        )


class FailingKnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Failing knowledge search"
    args_schema = KnowledgeArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False, error="retrieval failed")


class InvalidKnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Invalid knowledge search"
    args_schema = KnowledgeArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result="invalid")


class FakeCalculatorTool(BaseTool):
    name = "calculator"
    description = "Calculate"
    args_schema = CalculatorArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"value": 2})


class FakeEchoTool(BaseTool):
    name = "echo"
    description = "Echo"
    args_schema = EchoArgs

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"text": arguments.get("text")})


class ResearchPlannerStrategy:
    async def adecide(self, state):
        return AgentDecision(
            action="tool_calls",
            tool_calls=[
                AgentToolCall(
                    id="research_call",
                    tool_name="knowledge_search",
                    arguments={
                        "query": state["query"],
                        "knowledge_base_id": state.get("knowledge_base_id"),
                        "conversation_id": state.get("conversation_id"),
                        "memory_context": state.get("memory_context"),
                    },
                )
            ],
            metadata={
                "actual_strategy": "json_plan",
                "tool_scope": state.get("metadata", {}).get("tool_scope", {}),
            },
        )


class NoToolFinalPlannerStrategy:
    async def adecide(self, state):
        return AgentDecision(
            action="final",
            content="free form answer",
            metadata={"actual_strategy": "json_plan"},
        )


class CapturePlannerLLM:
    def __init__(self) -> None:
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        return type(
            "Resp",
            (),
            {
                "answer": '{"steps":[{"tool":"knowledge_search","args":{"query":"x"}},'
                '{"tool":"echo","args":{"text":"x"}}]}',
                "model": "fake",
            },
        )()


@pytest.fixture(autouse=True)
def reset_definitions():
    reset_agent_definition_registry()
    yield
    reset_agent_definition_registry()


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _copy_definition(base: AgentDefinition, **updates) -> AgentDefinition:
    data = base.model_dump(by_alias=True)
    data.update(updates)
    return AgentDefinition(**data)


def _run_with_planner(
    *,
    monkeypatch,
    planner,
    agent_id: str,
    registry: ToolRegistry | None = None,
    query: str = "query",
):
    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.get_planner_strategy",
        lambda state=None: planner,
    )
    graph = build_agent_graph(
        planner_node=PlannerNode(),
        tool_node=ToolNode(ToolExecutor(registry=registry or ToolRegistry())),
        final_node=FinalNode(),
        async_mode=True,
    )
    return asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(
            AgentRuntimeRequest(query=query, agent_id=agent_id, knowledge_base_id=7)
        )
    )


def test_knowledge_research_agent_registered_and_differs_from_general() -> None:
    registry = reset_agent_definition_registry()

    general = registry.get("general_agent")
    research = registry.get("knowledge_research_agent")

    assert research.retrieval_policy["enabled"] is True
    assert research.retrieval_policy["required"] is True
    assert research.instructions != general.instructions
    assert research.tool_allowlist != general.tool_allowlist
    assert research.output_mode == "grounded_research_answer"
    assert "echo" not in research.tool_allowlist
    assert {"knowledge_search", "calculator"} <= set(research.tool_allowlist)


def test_research_planner_only_sees_research_allowed_tools(monkeypatch) -> None:
    registry = _registry(FakeKnowledgeSearchTool(), FakeCalculatorTool(), FakeEchoTool())
    llm = CapturePlannerLLM()
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.get_tool_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: llm,
    )

    plan = LLMPlanner().plan(
        query="summarize",
        tool_allowlist=["knowledge_search", "calculator"],
        tool_scope_source="agent_definition",
    )

    assert [step.tool for step in plan.steps] == ["knowledge_search"]
    available = plan.metadata["tool_registry"]["available_tools"]
    assert available == ["calculator", "knowledge_search"]
    payload = llm.requests[0].messages[1].content
    assert "knowledge_search" in payload
    assert "calculator" in payload
    assert "echo" not in payload


def test_research_agent_answers_with_sources_citations_and_trace(monkeypatch) -> None:
    knowledge_tool = FakeKnowledgeSearchTool(with_evidence=True)
    registry = _registry(knowledge_tool, FakeCalculatorTool())
    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.get_planner_strategy",
        lambda state=None: ResearchPlannerStrategy(),
    )
    graph = build_agent_graph(
        planner_node=PlannerNode(),
        tool_node=ToolNode(ToolExecutor(registry=registry)),
        final_node=FinalNode(),
        async_mode=True,
    )

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(
            AgentRuntimeRequest(
                query="summarize current material",
                agent_id="knowledge_research_agent",
                knowledge_base_id=7,
            )
        )
    )

    assert result.answer == "基于证据的研究结论。"
    assert result.sources[0]["source"] == "policy-a.pdf"
    assert result.citations[0]["document_id"] == 101
    assert result.metadata["agent_id"] == "knowledge_research_agent"
    assert result.metadata["retrieval_required"] is True
    assert result.metadata["retrieval_used"] is True
    assert result.metadata["evidence_count"] == 1
    assert result.metadata["source_count"] == 1
    assert result.metadata["citation_count"] == 1
    assert result.metadata["selected_document_ids"] == [101]
    assert result.metadata["grounded_answer"] is True
    assert result.metadata["no_evidence"] is False
    assert result.metadata["tool_scope"]["tool_scope_source"] == "agent_definition"
    assert "tool_completed" in [step.step for step in result.trace]


def test_research_agent_refuses_when_no_evidence(monkeypatch) -> None:
    registry = _registry(FakeKnowledgeSearchTool(with_evidence=False), FakeCalculatorTool())
    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.get_planner_strategy",
        lambda state=None: ResearchPlannerStrategy(),
    )
    graph = build_agent_graph(
        planner_node=PlannerNode(),
        tool_node=ToolNode(ToolExecutor(registry=registry)),
        final_node=FinalNode(),
        async_mode=True,
    )

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=graph).arun(
            AgentRuntimeRequest(
                query="summarize unavailable material",
                agent_id="knowledge_research_agent",
                knowledge_base_id=7,
            )
        )
    )

    assert "无法基于当前知识库证据回答" in result.answer
    assert result.sources == []
    assert result.citations == []
    assert result.metadata["no_evidence"] is True
    assert result.metadata["grounded_answer"] is False


def test_enabled_without_required_does_not_force_refusal(monkeypatch) -> None:
    registry = get_agent_definition_registry()
    base = registry.get("general_agent")
    registry.register(
        _copy_definition(
            base,
            id="optional_retrieval_agent",
            retrieval_policy={"enabled": True, "required": False, "require_evidence": False},
        )
    )

    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=NoToolFinalPlannerStrategy(),
        agent_id="optional_retrieval_agent",
    )

    assert result.answer == "free form answer"
    assert result.metadata["retrieval_required"] is False
    assert result.metadata["no_evidence"] is False
    assert result.metadata["grounded_answer"] is False


def test_required_refuses_when_planner_does_not_retrieve(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=NoToolFinalPlannerStrategy(),
        agent_id="knowledge_research_agent",
    )

    assert result.answer == "无法基于当前知识库证据回答该问题。"
    assert result.metadata["retrieval_required"] is True
    assert result.metadata["retrieval_used"] is False
    assert result.metadata["no_evidence"] is True
    assert result.metadata["grounded_answer"] is False


def test_require_evidence_refuses_when_retrieval_returns_empty(monkeypatch) -> None:
    registry = get_agent_definition_registry()
    base = registry.get("general_agent")
    registry.register(
        _copy_definition(
            base,
            id="require_evidence_agent",
            tool_allowlist=["knowledge_search"],
            retrieval_policy={"enabled": True, "required": False, "require_evidence": True},
        )
    )

    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=ResearchPlannerStrategy(),
        agent_id="require_evidence_agent",
        registry=_registry(FakeKnowledgeSearchTool(with_evidence=False)),
    )

    assert result.answer == "无法基于当前知识库证据回答该问题。"
    assert result.metadata["retrieval_required"] is True
    assert result.metadata["retrieval_used"] is True
    assert result.metadata["no_evidence"] is True


def test_retrieval_failure_refuses_for_required_agent(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=ResearchPlannerStrategy(),
        agent_id="knowledge_research_agent",
        registry=_registry(FailingKnowledgeSearchTool()),
    )

    assert result.answer == "无法基于当前知识库证据回答该问题。"
    assert result.metadata["retrieval_used"] is False
    assert result.metadata["no_evidence"] is True
    assert result.metadata["grounded_answer"] is False


def test_invalid_retrieval_result_refuses_for_required_agent(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=ResearchPlannerStrategy(),
        agent_id="knowledge_research_agent",
        registry=_registry(InvalidKnowledgeSearchTool()),
    )

    assert result.answer == "无法基于当前知识库证据回答该问题。"
    assert result.metadata["retrieval_used"] is False
    assert result.metadata["no_evidence"] is True


def test_sources_only_counts_as_grounded(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=ResearchPlannerStrategy(),
        agent_id="knowledge_research_agent",
        registry=_registry(FakeKnowledgeSearchTool(evidence_mode="sources_only")),
    )

    assert result.answer == "基于证据的研究结论。"
    assert result.metadata["source_count"] == 1
    assert result.metadata["citation_count"] == 0
    assert result.metadata["grounded_answer"] is True
    assert result.metadata["no_evidence"] is False


def test_citations_only_counts_as_grounded(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=ResearchPlannerStrategy(),
        agent_id="knowledge_research_agent",
        registry=_registry(FakeKnowledgeSearchTool(evidence_mode="citations_only")),
    )

    assert result.answer == "基于证据的研究结论。"
    assert result.metadata["source_count"] == 0
    assert result.metadata["citation_count"] == 1
    assert result.metadata["grounded_answer"] is True
    assert result.metadata["no_evidence"] is False


def test_general_agent_is_not_forced_to_refuse_without_retrieval(monkeypatch) -> None:
    result = _run_with_planner(
        monkeypatch=monkeypatch,
        planner=NoToolFinalPlannerStrategy(),
        agent_id="general_agent",
    )

    assert result.answer == "free form answer"
    assert result.metadata["retrieval_required"] is False
    assert result.metadata["no_evidence"] is False


def test_research_agent_streaming_completes(monkeypatch) -> None:
    registry = _registry(FakeKnowledgeSearchTool(with_evidence=True), FakeCalculatorTool())
    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.get_planner_strategy",
        lambda state=None: ResearchPlannerStrategy(),
    )

    async def fake_collect_streaming_answer(request, on_delta):
        await on_delta("streamed research answer")
        return "streamed research answer"

    monkeypatch.setattr(
        "backend.app.agents.langgraph.nodes.collect_streaming_answer",
        fake_collect_streaming_answer,
    )
    graph = build_agent_graph(
        planner_node=PlannerNode(),
        tool_node=ToolNode(ToolExecutor(registry=registry)),
        final_node=FinalNode(),
        async_mode=True,
    )

    async def collect_events():
        runtime = LangGraphAgentRuntime(graph_app=graph)
        return [
            item
            async for item in runtime.astream_events(
                AgentRuntimeRequest(
                    query="compare current sources",
                    agent_id="knowledge_research_agent",
                    knowledge_base_id=7,
                )
            )
        ]

    events = asyncio.run(collect_events())

    assert "status" in [item["event"] for item in events]
    assert "answer_delta" in [item["event"] for item in events]
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["result"]["metadata"]["citation_count"] == 1


def test_research_agent_does_not_add_parallel_runtime_or_retriever() -> None:
    files = [
        path.name
        for path in Path("backend/app").rglob("*.py")
        if "knowledge_research" in path.name.lower()
    ]
    assert files == []

    code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("backend/app").rglob("*.py")
        if "agents" in path.parts or "tools" in path.parts
    )
    forbidden = [
        "KnowledgeResearchRuntime",
        "KnowledgeResearchRetriever",
        "KnowledgeResearchRAGService",
        "KnowledgeResearchCitationEngine",
        "劳动法",
        "财报",
        "合同",
        "员工手册",
        "贵州茅台",
        "归母净利润",
        "促进就业",
    ]
    assert not [token for token in forbidden if token in code]
