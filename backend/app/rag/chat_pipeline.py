from backend.app.chat.base import ChatSource, CitationItem
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.rag.schemas import RagChatInput, RagChatResult
from backend.app.retrievers.pipeline import RetrieverPipeline, RetrieverPipelineContext


class RagChatPipeline:
    def run(self, input: RagChatInput) -> RagChatResult:
        pipeline_context = RetrieverPipeline().run(
            RetrieverPipelineContext(
                query=input.query,
                knowledge_base_id=input.knowledge_base_id,
                top_k=input.top_k,
                score_threshold=input.score_threshold,
                metadata_filter=input.metadata_filter,
            )
        )
        context_text = pipeline_context.context_text
        context_chunks = pipeline_context.context_chunks
        base_metadata = self._build_metadata(input=input, context=pipeline_context)

        if not context_text.strip() or not context_chunks:
            metadata = {
                **base_metadata,
                "guardrail_triggered": True,
                "guardrail_reason": "empty_context",
                "llm_called": False,
            }
            return RagChatResult(
                answer="根据当前知识库内容无法回答该问题。",
                sources=[],
                citations=[],
                context_text="",
                prompt_text="",
                llm_model=None,
                metadata=metadata,
            )

        prompt_result = PromptBuilderFactory.get_builder().build(
            PromptBuildRequest(
                query=input.query,
                context_text=context_text,
            )
        )
        prompt_messages = [
            LLMMessage(role=message.role, content=message.content)
            for message in prompt_result.messages
        ]
        prompt_messages = self._inject_memory_messages(
            prompt_messages=prompt_messages,
            memory_messages=input.memory_messages,
        )
        llm_response = LLMFactory.get_llm().chat(
            LLMRequest(
                messages=prompt_messages,
                model=input.model,
                metadata={
                    "rag_chat_pipeline": True,
                },
            )
        )

        metadata = {
            **base_metadata,
            "guardrail_triggered": False,
            "llm_called": True,
            "llm_usage": llm_response.usage,
            "llm_metadata": llm_response.metadata,
        }
        return RagChatResult(
            answer=llm_response.answer,
            sources=self._build_sources(context_chunks),
            citations=self._build_citations(context_chunks),
            context_text=context_text,
            prompt_text=prompt_result.prompt_text,
            llm_model=llm_response.model,
            metadata=metadata,
        )

    def _build_metadata(
        self,
        input: RagChatInput,
        context: RetrieverPipelineContext,
    ) -> dict:
        pipeline_metadata = dict(context.metadata)
        query_rewrite = pipeline_metadata.get("query_rewrite", {})
        original_query = context.original_query or input.query
        rewritten_query = context.rewritten_query or context.active_query
        context_compression = pipeline_metadata.get("context_compression", {})

        metadata = {
            **pipeline_metadata,
            "retrieved_total": len(context.fused_chunks),
            "reranked_total": len(context.reranked_chunks),
            "context_total_chunks": len(context.context_chunks),
            "context_total_chars": len(context.context_text),
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "query_rewrite_changed": bool(
                query_rewrite.get("changed", original_query != rewritten_query)
            ),
            "context_compression_applied": self._compression_applied(
                context_compression
            ),
            "context_original_chars": context_compression.get(
                "original_chars", len(context.context_text)
            ),
            "context_compressed_chars": context_compression.get(
                "compressed_chars", len(context.context_text)
            ),
            "context_original_chunks": context_compression.get(
                "original_chunk_count", len(context.context_chunks)
            ),
            "context_compressed_chunks": context_compression.get(
                "compressed_chunk_count", len(context.context_chunks)
            ),
            "context_compression": context_compression,
            "metadata_filter": input.metadata_filter,
            "metadata_filter_applied": bool(input.metadata_filter),
            "reranker": pipeline_metadata.get("reranker", {}),
            "mmr": pipeline_metadata.get("mmr", {}),
            "neighbor_expansion": pipeline_metadata.get("neighbor_expansion", {}),
            "errors": context.errors or pipeline_metadata.get("errors", []),
        }
        return metadata

    def _inject_memory_messages(
        self,
        prompt_messages: list[LLMMessage],
        memory_messages: list[LLMMessage],
    ) -> list[LLMMessage]:
        if not memory_messages:
            return prompt_messages

        insert_index = 1 if prompt_messages and prompt_messages[0].role == "system" else 0
        return [
            *prompt_messages[:insert_index],
            *memory_messages,
            *prompt_messages[insert_index:],
        ]

    def _build_sources(self, chunks: list) -> list[ChatSource]:
        return [
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
            for chunk in chunks
        ]

    def _build_citations(self, chunks: list) -> list[CitationItem]:
        return [
            CitationItem(
                source=chunk.source,
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                text_preview=chunk.text[:120] if chunk.text else None,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

    def _compression_applied(self, metadata: dict) -> bool:
        if not metadata.get("enabled") or metadata.get("failed"):
            return False
        return bool(
            metadata.get("compressed_chars", 0) < metadata.get("original_chars", 0)
            or metadata.get("compressed_chunk_count", 0)
            < metadata.get("original_chunk_count", 0)
        )
