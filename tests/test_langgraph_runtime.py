from backend.app.agents import AgentRuntime
from backend.app.agents.langgraph import AgentRuntimeFactory, LangGraphAgentRuntime, LLMPlanner
from backend.app.agents.langgraph.nodes import ToolNode
from backend.app.agents.langgraph.state import create_initial_state
from backend.app.agents.state import AgentRuntimeRequest
from backend.app.llms import LLMResponse
from backend.app.rag import RagChatResult
from backend.app.tools import ToolResult
from backend.app.tools.builtin.knowledge_tool import KnowledgeSearchTool


class FakePlannerLLM:
    def chat(self, request):
        return LLMResponse(
            answer='{"steps":[{"tool":"knowledge_search","args":{"query":"劳动法"}}]}',
            model="fake-planner",
        )


class FakeToolExecutor:
    def __init__(self) -> None:
        self.called = False
        self.last_call = None

    def execute(self, tool_call):
        self.called = True
        self.last_call = tool_call
        return ToolResult(
            name=tool_call.name,
            success=True,
            result={
                "answer": "第二章讲促进就业。",
                "sources": [{"source": "中国劳动法.pdf"}],
                "citations": [{"source": "中国劳动法.pdf"}],
                "metadata": {"retriever_mode": "hybrid"},
            },
        )


class FakeGraphApp:
    def invoke(self, state):
        state["tool_calls"].append(
            {
                "name": "knowledge_search",
                "arguments": {"query": state["query"]},
            }
        )
        state["tool_results"].append(
            {
                "name": "knowledge_search",
                "success": True,
                "result": {
                    "answer": "graph answer",
                    "sources": [{"source": "source.pdf"}],
                    "citations": [{"source": "source.pdf"}],
                    "metadata": {"runtime": "langgraph"},
                },
                "error": None,
                "metadata": {},
            }
        )
        state["knowledge"] = state["tool_results"][0]["result"]
        state["final_answer"] = "graph answer"
        state["metadata"]["trace"].append(
            {
                "step": "final_answer",
                "name": "final_answer",
                "input": {"query": state["query"]},
                "output": {"answer": "graph answer"},
                "duration_ms": 0,
                "status": "success",
                "error": None,
            }
        )
        return state


class FakeRagPipeline:
    def run(self, input):
        return RagChatResult(
            answer="knowledge answer",
            sources=[],
            citations=[],
            context_text="context",
            prompt_text="prompt",
            llm_model="fake",
            metadata={"retriever_mode": "hybrid"},
        )


def test_factory_creates_langgraph_runtime():
    runtime = AgentRuntimeFactory.get_runtime("langgraph")

    assert isinstance(runtime, LangGraphAgentRuntime)


def test_factory_defaults_to_v1(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.langgraph.factory.settings.AGENT_RUNTIME",
        "v1",
    )

    runtime = AgentRuntimeFactory.get_runtime()

    assert isinstance(runtime, AgentRuntime)


def test_planner_returns_plan(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.langgraph.planner.LLMFactory.get_llm",
        lambda: FakePlannerLLM(),
    )

    plan = LLMPlanner().plan(query="劳动法第二章说什么", knowledge_base_id=4)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "knowledge_search"
    assert plan.steps[0].args["query"] == "劳动法"
    assert plan.steps[0].args["knowledge_base_id"] == 4


def test_tool_node_calls_tool_executor():
    executor = FakeToolExecutor()
    state = create_initial_state(
        query="劳动法第二章说什么",
        conversation_id=None,
        knowledge_base_id=4,
        memory_context=None,
        metadata={},
    )
    state["plan"] = {
        "steps": [
            {
                "tool": "knowledge_search",
                "args": {"query": "劳动法第二章说什么", "knowledge_base_id": 4},
            }
        ]
    }

    result_state = ToolNode(tool_executor=executor)(state)

    assert executor.called is True
    assert executor.last_call.name == "knowledge_search"
    assert result_state["knowledge"]["answer"] == "第二章讲促进就业。"


def test_knowledge_tool_works(monkeypatch):
    monkeypatch.setattr(
        "backend.app.tools.builtin.knowledge_tool.RagChatPipeline",
        FakeRagPipeline,
    )

    result = KnowledgeSearchTool().run({"query": "劳动法", "knowledge_base_id": 4})

    assert result.success is True
    assert isinstance(result.result, dict)
    assert result.result["answer"] == "knowledge answer"


def test_langgraph_runtime_returns_final_answer():
    result = LangGraphAgentRuntime(graph_app=FakeGraphApp()).run(
        AgentRuntimeRequest(query="劳动法第二章说什么", knowledge_base_id=4)
    )

    assert result.answer == "graph answer"
    assert result.action == "tool"
    assert result.sources[0]["source"] == "source.pdf"
    assert result.trace[0].step == "final_answer"


def test_config_switch_to_langgraph(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.langgraph.factory.settings.AGENT_RUNTIME",
        "langgraph",
    )

    runtime = AgentRuntimeFactory.get_runtime()

    assert isinstance(runtime, LangGraphAgentRuntime)


def test_v1_runtime_not_affected(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.runtime.LLMFactory.get_llm",
        lambda: FakePlannerLLM(),
    )

    result = AgentRuntime().run(AgentRuntimeRequest(query="你好"))

    assert result.action in {"direct_answer", "tool"}
    assert result.trace
