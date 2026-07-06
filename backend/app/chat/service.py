from backend.app.chat.base import ChatRequest, ChatResponse, ChatSource, CitationItem
from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.context.compression import SimpleContextCompressor
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.query import SimpleQueryRewriter
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery


class ChatService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        rewrite_result = SimpleQueryRewriter().rewrite(request.query)

        retriever = RetrieverFactory.get_hybrid_retriever()
        retrieve_result = retriever.retrieve(
            HybridRetrieveQuery(
                query=rewrite_result.rewritten_query,
                knowledge_base_id=request.knowledge_base_id,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )
        )

        reranker = RerankerFactory.get_reranker()
        rerank_result = reranker.rerank(
            RerankQuery(
                query=rewrite_result.rewritten_query,
                chunks=retrieve_result.chunks,
                top_k=request.top_k,
            )
        )

        context_builder = ContextBuilderFactory.get_builder()
        context_result = context_builder.build(
            ContextBuildRequest(
                query=rewrite_result.rewritten_query,
                chunks=rerank_result.chunks,
            )
        )

        compression_result = SimpleContextCompressor().compress(
            context_text=context_result.context_text,
            chunks=context_result.chunks,
        )

        if not compression_result.context_text.strip() or not compression_result.chunks:
            return ChatResponse(
                query=request.query,
                answer="根据当前知识库内容无法回答该问题。",
                sources=[],
                citations=[],
                context_text="",
                prompt_text="",
                llm_model=None,
                metadata={
                    "retrieved_total": retrieve_result.total,
                    "reranked_total": rerank_result.total,
                    "context_total_chunks": context_result.total_chunks,
                    "context_total_chars": context_result.total_chars,
                    "original_query": rewrite_result.original_query,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "query_rewrite_changed": rewrite_result.changed,
                    "context_compression_applied": compression_result.compression_applied,
                    "context_original_chars": compression_result.original_chars,
                    "context_compressed_chars": compression_result.compressed_chars,
                    "context_original_chunks": compression_result.metadata.get(
                        "original_chunk_count", 0
                    ),
                    "context_compressed_chunks": compression_result.metadata.get(
                        "compressed_chunk_count", 0
                    ),
                    "metadata_filter": request.metadata_filter,
                    "metadata_filter_applied": bool(request.metadata_filter),
                    "guardrail_triggered": True,
                    "guardrail_reason": "empty_context",
                    "llm_called": False,
                    **retrieve_result.metadata,
                },
            )

        prompt_builder = PromptBuilderFactory.get_builder()
        prompt_result = prompt_builder.build(
            PromptBuildRequest(
                query=request.query,
                context_text=compression_result.context_text,
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
                for chunk in compression_result.chunks
            ],
            citations=[
                CitationItem(
                    source=chunk.source,
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    text_preview=chunk.text[:120] if chunk.text else None,
                    metadata=chunk.metadata,
                )
                for chunk in compression_result.chunks
            ],
            context_text=compression_result.context_text,
            prompt_text=prompt_result.prompt_text,
            llm_model=llm_response.model,
            metadata={
                "retrieved_total": retrieve_result.total,
                "reranked_total": rerank_result.total,
                "context_total_chunks": context_result.total_chunks,
                "context_total_chars": context_result.total_chars,
                "original_query": rewrite_result.original_query,
                "rewritten_query": rewrite_result.rewritten_query,
                "query_rewrite_changed": rewrite_result.changed,
                "context_compression_applied": compression_result.compression_applied,
                "context_original_chars": compression_result.original_chars,
                "context_compressed_chars": compression_result.compressed_chars,
                "context_original_chunks": compression_result.metadata.get(
                    "original_chunk_count", 0
                ),
                "context_compressed_chunks": compression_result.metadata.get(
                    "compressed_chunk_count", 0
                ),
                "metadata_filter": request.metadata_filter,
                "metadata_filter_applied": bool(request.metadata_filter),
                "guardrail_triggered": False,
                "llm_called": True,
                "llm_usage": llm_response.usage,
                **retrieve_result.metadata,
            },
        )
