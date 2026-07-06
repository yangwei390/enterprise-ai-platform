from backend.app.chat.base import ChatRequest, ChatResponse, ChatSource
from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieveQuery, RetrieverFactory


class ChatService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        retriever = RetrieverFactory.get_retriever()
        retrieve_result = retriever.retrieve(
            RetrieveQuery(
                query=request.query,
                knowledge_base_id=request.knowledge_base_id,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )
        )

        reranker = RerankerFactory.get_reranker()
        rerank_result = reranker.rerank(
            RerankQuery(
                query=request.query,
                chunks=retrieve_result.chunks,
                top_k=request.top_k,
            )
        )

        context_builder = ContextBuilderFactory.get_builder()
        context_result = context_builder.build(
            ContextBuildRequest(
                query=request.query,
                chunks=rerank_result.chunks,
            )
        )

        if not context_result.context_text.strip() or not context_result.chunks:
            return ChatResponse(
                query=request.query,
                answer="根据当前知识库内容无法回答该问题。",
                sources=[],
                context_text="",
                prompt_text="",
                llm_model=None,
                metadata={
                    "retrieved_total": retrieve_result.total,
                    "reranked_total": rerank_result.total,
                    "context_total_chunks": context_result.total_chunks,
                    "context_total_chars": context_result.total_chars,
                    "metadata_filter": request.metadata_filter,
                    "metadata_filter_applied": bool(request.metadata_filter),
                    "guardrail_triggered": True,
                    "guardrail_reason": "empty_context",
                    "llm_called": False,
                },
            )

        prompt_builder = PromptBuilderFactory.get_builder()
        prompt_result = prompt_builder.build(
            PromptBuildRequest(
                query=request.query,
                context_text=context_result.context_text,
            )
        )

        llm = LLMFactory.get_llm()
        llm_response = llm.chat(
            LLMRequest(
                messages=[
                    LLMMessage(role=message.role, content=message.content)
                    for message in prompt_result.messages
                ]
            )
        )

        return ChatResponse(
            query=request.query,
            answer=llm_response.answer,
            sources=[
                ChatSource(
                    id=chunk.id,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
                for chunk in context_result.chunks
            ],
            context_text=context_result.context_text,
            prompt_text=prompt_result.prompt_text,
            llm_model=llm_response.model,
            metadata={
                "retrieved_total": retrieve_result.total,
                "reranked_total": rerank_result.total,
                "context_total_chunks": context_result.total_chunks,
                "context_total_chars": context_result.total_chars,
                "metadata_filter": request.metadata_filter,
                "metadata_filter_applied": bool(request.metadata_filter),
                "guardrail_triggered": False,
                "llm_called": True,
                "llm_usage": llm_response.usage,
            },
        )
