import pytest
from backend.app.chat import ChatRequest, ChatService
from backend.app.chat.base import ChatSource, CitationItem
from backend.app.context import ContextChunk
from backend.app.llms import LLMMessage, LLMResponse
from backend.app.prompts import PromptBuildResult, PromptMessage
from backend.app.rag import RagChatInput, RagChatPipeline, RagChatResult
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.pipeline import (
    BaseRetrieverStep,
    RetrieverPipeline,
    RetrieverPipelineContext,
)


class FakeRetrieverPipeline:
    called = False

    def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
        FakeRetrieverPipeline.called = True
        context.original_query = context.query
        context.rewritten_query = "劳动法第二章说什么"
        context.fused_chunks = [
            RetrievedChunk(
                id="chunk-1",
                score=0.9,
                text="fused text",
                document_id=10,
                knowledge_base_id=20,
                chunk_index=1,
                metadata={"source": "law.pdf"},
            )
        ]
        context.reranked_chunks = [
            RerankedChunk(
                id="chunk-1",
                original_score=0.9,
                rerank_score=0.95,
                text="reranked text",
                document_id=10,
                knowledge_base_id=20,
                chunk_index=1,
                metadata={
                    "source": "law.pdf",
                    "rerank_score": 0.95,
                    "rerank_rank": 1,
                },
            )
        ]
        context.context_text = "PIPELINE CONTEXT"
        context.context_chunks = [
            ContextChunk(
                id="chunk-1",
                text="劳动法第二章促进就业",
                document_id=10,
                knowledge_base_id=20,
                chunk_index=1,
                score=0.95,
                source="law.pdf",
                metadata={
                    "source": "law.pdf",
                    "rerank_score": 0.95,
                    "rerank_rank": 1,
                },
            )
        ]
        context.metadata = {
            "query_rewrite": {
                "original_query": context.query,
                "rewritten_query": context.rewritten_query,
                "changed": True,
            },
            "retriever_mode": "hybrid",
            "fusion": "rrf",
            "soft_boost_applied": False,
            "reranker": {
                "rerank_enabled": True,
                "rerank_provider": "fake",
            },
            "mmr": {
                "enabled": True,
                "selected_chunk_count": 1,
            },
            "neighbor_expansion": {
                "enabled": True,
                "added_chunk_count": 0,
            },
            "context_compression": {
                "enabled": True,
                "provider": "fake",
                "failed": False,
                "original_chars": 20,
                "compressed_chars": 16,
                "original_chunk_count": 1,
                "compressed_chunk_count": 1,
            },
            "auto_filter_applied": True,
            "soft_boost": {"applied": False},
            "errors": [],
        }
        return context


class FakePromptBuilder:
    last_context_text: str | None = None

    def build(self, request):
        FakePromptBuilder.last_context_text = request.context_text
        return PromptBuildResult(
            messages=[PromptMessage(role="user", content=request.query)],
            prompt_text=f"prompt: {request.context_text}",
            metadata={},
        )


class FakeLLM:
    def __init__(self) -> None:
        self.messages: list[LLMMessage] = []

    def chat(self, request):
        self.messages = request.messages
        return LLMResponse(
            answer="第二章讲促进就业。",
            model="fake-llm",
            usage={},
            metadata={"provider": "fake"},
        )


class FakeRagChatPipeline:
    called = False

    def run(self, input):
        FakeRagChatPipeline.called = True
        return RagChatResult(
            answer="fake answer",
            sources=[
                ChatSource(
                    id="chunk-1",
                    text="source text",
                    document_id=1,
                    knowledge_base_id=2,
                    chunk_index=3,
                    score=0.9,
                    source="source.txt",
                    metadata={},
                )
            ],
            citations=[
                CitationItem(
                    source="source.txt",
                    document_id=1,
                    knowledge_base_id=2,
                    chunk_index=3,
                    score=0.9,
                    text_preview="source text",
                    metadata={},
                )
            ],
            context_text="context",
            prompt_text="prompt",
            llm_model="fake-llm",
            metadata={
                "retrieved_total": 1,
                "reranker": {},
                "mmr": {},
                "neighbor_expansion": {},
                "context_compression": {},
            },
        )


def _patch_pipeline_dependencies(monkeypatch, fake_llm: FakeLLM | None = None) -> FakeLLM:
    llm = fake_llm or FakeLLM()
    FakeRetrieverPipeline.called = False
    FakePromptBuilder.last_context_text = None
    monkeypatch.setattr("backend.app.rag.chat_pipeline.RetrieverPipeline", FakeRetrieverPipeline)
    monkeypatch.setattr(
        "backend.app.rag.chat_pipeline.PromptBuilderFactory.get_builder",
        lambda: FakePromptBuilder(),
    )
    monkeypatch.setattr("backend.app.rag.chat_pipeline.LLMFactory.get_llm", lambda: llm)
    return llm


def test_rag_chat_pipeline_uses_retriever_pipeline(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)

    result = RagChatPipeline().run(
        RagChatInput(
            query="请问劳动法第二章说什么",
            knowledge_base_id=20,
        )
    )

    assert FakeRetrieverPipeline.called is True
    assert result.answer == "第二章讲促进就业。"
    assert result.sources[0].source == "law.pdf"
    assert result.metadata["retrieved_total"] == 1


def test_rag_chat_pipeline_uses_pipeline_context_text(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)

    result = RagChatPipeline().run(RagChatInput(query="测试问题"))

    assert FakePromptBuilder.last_context_text == "PIPELINE CONTEXT"
    assert result.context_text == "PIPELINE CONTEXT"
    assert result.prompt_text == "prompt: PIPELINE CONTEXT"


def test_rag_chat_pipeline_preserves_pipeline_metadata(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)

    result = RagChatPipeline().run(RagChatInput(query="测试问题"))

    assert result.metadata["reranker"]["rerank_provider"] == "fake"
    assert result.metadata["mmr"]["enabled"] is True
    assert result.metadata["neighbor_expansion"]["enabled"] is True
    assert result.metadata["context_compression"]["provider"] == "fake"
    assert result.metadata["fusion"] == "rrf"
    assert result.metadata["soft_boost_applied"] is False
    assert result.metadata["errors"] == []


def test_rag_chat_pipeline_does_not_save_messages(monkeypatch):
    fake_llm = _patch_pipeline_dependencies(monkeypatch)

    result = RagChatPipeline().run(
        RagChatInput(
            query="测试问题",
            memory_messages=[LLMMessage(role="system", content="历史记忆")],
        )
    )

    assert result.answer == "第二章讲促进就业。"
    assert fake_llm.messages[0].content == "历史记忆"
    assert not hasattr(RagChatPipeline(), "conversation_service")


def test_chat_service_response_unchanged_after_unify(monkeypatch):
    FakeRagChatPipeline.called = False
    monkeypatch.setattr("backend.app.chat.service.RagChatPipeline", FakeRagChatPipeline)

    response = ChatService().chat(
        ChatRequest(
            query="测试问题",
            enable_tools=False,
        )
    )

    assert FakeRagChatPipeline.called is True
    assert response.answer == "fake answer"
    assert response.sources[0].source == "source.txt"
    assert response.citations[0].source == "source.txt"
    assert response.context_text == "context"
    assert response.prompt_text == "prompt"
    assert response.metadata["tools_enabled"] is False


def test_retriever_pipeline_result_contains_context_fields():
    class FakeContextStep(BaseRetrieverStep):
        def run(self, context: RetrieverPipelineContext) -> RetrieverPipelineContext:
            context.context_text = "pipeline context"
            context.context_chunks = [
                ContextChunk(
                    id="chunk-1",
                    text="chunk text",
                    document_id=1,
                    knowledge_base_id=2,
                    chunk_index=3,
                    score=0.8,
                    source="source.txt",
                    metadata={"source": "source.txt"},
                )
            ]
            context.metadata["context_compression"] = {"enabled": False}
            return context

    result = RetrieverPipeline(steps=[FakeContextStep()]).run(
        RetrieverPipelineContext(query="测试问题")
    )

    assert result.context_text == "pipeline context"
    assert result.context_chunks[0].source == "source.txt"
    assert result.metadata["context_compression"]["enabled"] is False


def test_rag_chat_pipeline_does_not_use_legacy_manual_components(monkeypatch):
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        "backend.app.rerankers.factory.RerankerFactory.get_reranker",
        lambda: pytest.fail("RagChatPipeline must not call RerankerFactory directly"),
    )

    RagChatPipeline().run(RagChatInput(query="测试问题"))
