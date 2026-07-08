from backend.app.agents import AgentRuntime, AgentRuntimeRequest
from backend.app.llms import LLMResponse
from backend.app.tools import ToolResult


class FakeToolExecutor:
    def __init__(self, result: ToolResult | None = None, should_raise: bool = False) -> None:
        self.result = result
        self.should_raise = should_raise
        self.called = False
        self.last_call = None

    def execute(self, tool_call):
        self.called = True
        self.last_call = tool_call
        if self.should_raise:
            raise RuntimeError("tool boom")
        if self.result is not None:
            return self.result
        return ToolResult(
            name=tool_call.name,
            success=True,
            result={"answer": "tool answer"},
        )


class FakeLLM:
    def chat(self, request):
        return LLMResponse(
            answer="direct llm answer",
            model="fake-llm",
            usage={},
            metadata={"provider": "fake"},
        )


def test_agent_runtime_routes_to_knowledge_tool():
    executor = FakeToolExecutor(
        ToolResult(
            name="knowledge_search",
            success=True,
            result={
                "answer": "第二章讲促进就业。",
                "sources": [{"source": "中国劳动法.pdf"}],
                "citations": [{"source": "中国劳动法.pdf"}],
                "metadata": {"retriever_mode": "hybrid"},
            },
        )
    )

    result = AgentRuntime(tool_executor=executor).run(
        AgentRuntimeRequest(query="劳动法第二章说什么", knowledge_base_id=4)
    )

    assert executor.called is True
    assert executor.last_call is not None
    assert executor.last_call.name == "knowledge_search"
    assert executor.last_call.arguments["query"] == "劳动法第二章说什么"
    assert executor.last_call.arguments["knowledge_base_id"] == 4
    assert result.action == "tool"
    assert result.answer == "第二章讲促进就业。"
    assert result.sources[0]["source"] == "中国劳动法.pdf"
    assert result.metadata["knowledge_metadata"]["retriever_mode"] == "hybrid"


def test_agent_runtime_routes_to_current_time():
    executor = FakeToolExecutor(
        ToolResult(
            name="get_current_time",
            success=True,
            result={"time": "2026-07-08T12:00:00+08:00"},
        )
    )

    result = AgentRuntime(tool_executor=executor).run(
        AgentRuntimeRequest(query="现在几点")
    )

    assert executor.called is True
    assert executor.last_call is not None
    assert executor.last_call.name == "get_current_time"
    assert result.answer == "当前时间是：2026-07-08T12:00:00+08:00"


def test_agent_runtime_routes_to_calculator():
    executor = FakeToolExecutor(
        ToolResult(
            name="calculator",
            success=True,
            result={"value": 3},
        )
    )

    result = AgentRuntime(tool_executor=executor).run(
        AgentRuntimeRequest(query="1+2 等于多少")
    )

    assert executor.called is True
    assert executor.last_call is not None
    assert executor.last_call.name == "calculator"
    assert executor.last_call.arguments["expression"] == "1+2"
    assert result.answer == "计算结果是：3"


def test_agent_runtime_direct_answer_when_no_tool_needed(monkeypatch):
    executor = FakeToolExecutor()
    monkeypatch.setattr("backend.app.agents.runtime.LLMFactory.get_llm", lambda: FakeLLM())

    result = AgentRuntime(tool_executor=executor).run(AgentRuntimeRequest(query="你好"))

    assert executor.called is False
    assert result.action == "direct_answer"
    assert result.answer == "direct llm answer"


def test_agent_runtime_returns_trace():
    executor = FakeToolExecutor(
        ToolResult(
            name="calculator",
            success=True,
            result={"value": 7},
        )
    )

    result = AgentRuntime(tool_executor=executor).run(
        AgentRuntimeRequest(query="1+2*3")
    )

    steps = [trace.step for trace in result.trace]
    assert "planner" in steps
    assert "tool_call" in steps
    assert "observation" in steps
    assert "final_answer" in steps


def test_agent_runtime_handles_tool_error():
    executor = FakeToolExecutor(should_raise=True)

    result = AgentRuntime(tool_executor=executor).run(
        AgentRuntimeRequest(query="劳动法第二章说什么")
    )

    assert executor.called is True
    assert result.answer == "工具调用失败：tool boom"
    assert any(trace.status == "failed" for trace in result.trace)
    assert any("tool boom" in error for error in result.metadata["errors"])
