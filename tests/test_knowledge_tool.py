from backend.app.chat.base import ChatSource, CitationItem
from backend.app.rag import RagChatResult
from backend.app.tools import ToolCall, ToolExecutor, get_tool_registry
from backend.app.tools.builtin.knowledge_tool import KnowledgeSearchTool
from backend.app.tools.registry import ToolRegistry


class FakeRagChatPipeline:
    called = False
    last_query: str | None = None
    last_knowledge_base_id: int | None = None
    last_conversation_id: int | None = None
    last_memory_context: str | None = None

    def run(self, input):
        FakeRagChatPipeline.called = True
        FakeRagChatPipeline.last_query = input.query
        FakeRagChatPipeline.last_knowledge_base_id = input.knowledge_base_id
        FakeRagChatPipeline.last_conversation_id = input.conversation_id
        FakeRagChatPipeline.last_memory_context = input.memory_context
        return RagChatResult(
            answer="知识库回答",
            sources=[
                ChatSource(
                    id="chunk-1",
                    text="source text",
                    document_id=1,
                    knowledge_base_id=2,
                    chunk_index=3,
                    score=0.9,
                    source="source.pdf",
                    metadata={"rerank_score": 0.95},
                )
            ],
            citations=[
                CitationItem(
                    source="source.pdf",
                    document_id=1,
                    knowledge_base_id=2,
                    chunk_index=3,
                    score=0.9,
                    text_preview="source text",
                    metadata={"rerank_score": 0.95},
                )
            ],
            context_text="context",
            prompt_text="prompt",
            llm_model="fake-llm",
            metadata={"retriever_mode": "hybrid"},
        )


def test_knowledge_tool_calls_rag_chat_pipeline(monkeypatch):
    FakeRagChatPipeline.called = False
    monkeypatch.setattr(
        "backend.app.tools.builtin.knowledge_tool.RagChatPipeline",
        FakeRagChatPipeline,
    )

    result = KnowledgeSearchTool().run(
        {
            "query": "劳动法第二章说什么",
            "knowledge_base_id": 2,
            "conversation_id": 10,
            "memory_context": "历史上下文",
        }
    )

    assert result.success is True
    assert FakeRagChatPipeline.called is True
    assert FakeRagChatPipeline.last_query == "劳动法第二章说什么"
    assert FakeRagChatPipeline.last_knowledge_base_id == 2
    assert FakeRagChatPipeline.last_conversation_id == 10
    assert FakeRagChatPipeline.last_memory_context == "历史上下文"


def test_knowledge_tool_returns_answer_sources_metadata(monkeypatch):
    monkeypatch.setattr(
        "backend.app.tools.builtin.knowledge_tool.RagChatPipeline",
        FakeRagChatPipeline,
    )

    result = KnowledgeSearchTool().run({"query": "测试问题"})

    assert result.success is True
    assert isinstance(result.result, dict)
    assert result.result["answer"] == "知识库回答"
    assert result.result["sources"][0]["source"] == "source.pdf"
    assert result.result["citations"][0]["chunk_index"] == 3
    assert result.result["metadata"]["retriever_mode"] == "hybrid"


def test_knowledge_tool_registered_in_registry(monkeypatch):
    import backend.app.tools.registry as registry_module

    monkeypatch.setattr(registry_module, "_tool_registry", None)

    tool = get_tool_registry().get_tool("knowledge_search")

    assert tool is not None
    assert tool.name == "knowledge_search"


def test_knowledge_tool_does_not_save_conversation_message(monkeypatch):
    monkeypatch.setattr(
        "backend.app.tools.builtin.knowledge_tool.RagChatPipeline",
        FakeRagChatPipeline,
    )

    result = KnowledgeSearchTool().run(
        {
            "query": "测试问题",
            "conversation_id": 1,
        }
    )

    assert result.success is True
    assert not hasattr(KnowledgeSearchTool(), "conversation_service")


def test_existing_tools_still_work():
    registry = ToolRegistry()
    for tool in get_tool_registry().list_tools():
        if tool.name in {"calculator", "echo", "get_current_time"}:
            registry.register(tool)
    executor = ToolExecutor(registry=registry)

    calculator_result = executor.execute(
        ToolCall(name="calculator", arguments={"expression": "1 + 2 * 3"})
    )
    echo_result = executor.execute(ToolCall(name="echo", arguments={"text": "hello"}))
    time_result = executor.execute(
        ToolCall(name="get_current_time", arguments={"timezone": "UTC"})
    )

    assert calculator_result.success is True
    assert calculator_result.result == {"value": 7}
    assert echo_result.success is True
    assert echo_result.result == {"text": "hello"}
    assert time_result.success is True
    assert isinstance(time_result.result, dict)
    assert "time" in time_result.result
