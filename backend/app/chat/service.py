from backend.app.chat.base import ChatRequest, ChatResponse, ChatSource, CitationItem
from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.context.compression import SimpleContextCompressor
from backend.app.conversations import ConversationService
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.logger import logger
from backend.app.models import Message
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.query import SimpleQueryRewriter
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery
from backend.app.schemas.conversation import ConversationCreate
from backend.app.tools import ToolCall, ToolExecutor, get_tool_registry


class ChatService:
    def __init__(self, conversation_service: ConversationService | None = None) -> None:
        self.conversation_service = conversation_service

    def chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id, history_loaded_count = self._prepare_conversation(request)

        rewrite_result = SimpleQueryRewriter().rewrite(request.query)

        if self.conversation_service is not None and conversation_id is not None:
            self.conversation_service.add_user_message(
                conversation_id=conversation_id,
                content=request.query,
                metadata={
                    "original_query": rewrite_result.original_query,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "query_rewrite_changed": rewrite_result.changed,
                },
            )

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
            answer = "根据当前知识库内容无法回答该问题。"
            metadata = {
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
                "tools_enabled": request.enable_tools,
                "tool_results": [],
                "history_loaded_count": history_loaded_count,
                **retrieve_result.metadata,
            }
            assistant_message = self._save_assistant_message(
                conversation_id=conversation_id,
                content=answer,
                metadata=metadata,
            )
            return ChatResponse(
                query=request.query,
                answer=answer,
                conversation_id=conversation_id,
                message_id=assistant_message.id if assistant_message else None,
                sources=[],
                citations=[],
                context_text="",
                prompt_text="",
                llm_model=None,
                metadata=metadata,
            )

        prompt_builder = PromptBuilderFactory.get_builder()
        prompt_result = prompt_builder.build(
            PromptBuildRequest(
                query=request.query,
                context_text=compression_result.context_text,
            )
        )

        function_tools: list[dict] = []
        if request.enable_tools:
            tool_definitions = get_tool_registry().get_tool_definitions()
            function_tools = [
                self._to_function_tool(tool_definition.model_dump())
                for tool_definition in tool_definitions
            ]

        llm = LLMFactory.get_llm()
        prompt_messages = [
            LLMMessage(role=message.role, content=message.content)
            for message in prompt_result.messages
        ]
        llm_request = LLMRequest(
            messages=prompt_messages,
            tools=function_tools if request.enable_tools else [],
            tool_choice="auto" if request.enable_tools and function_tools else None,
            metadata={
                "tools_enabled": request.enable_tools,
                "tools": function_tools,
            },
        )
        llm_response = self._call_llm_with_tool_fallback(llm, llm_request)

        sources = [
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
        ]
        citations = [
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
        ]
        metadata = {
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
            "llm_metadata": llm_response.metadata,
            "tools_enabled": request.enable_tools,
            "tool_call_count": len(llm_response.tool_calls),
            "tool_summary_llm_called": False,
            "tool_results": [],
            "history_loaded_count": history_loaded_count,
            **retrieve_result.metadata,
        }

        answer = llm_response.answer
        if request.enable_tools and llm_response.tool_calls:
            limited_tool_calls = llm_response.tool_calls[:3]
            tool_results = [
                ToolExecutor()
                .execute(
                    ToolCall(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )
                .model_dump()
                for tool_call in limited_tool_calls
            ]
            metadata["tool_call_count"] = len(limited_tool_calls)
            metadata["tool_results"] = tool_results
            answer, summary_called = self._summarize_tool_results(
                llm=llm,
                prompt_messages=prompt_messages,
                tool_results=tool_results,
            )
            metadata["tool_summary_llm_called"] = summary_called

        assistant_message = self._save_assistant_message(
            conversation_id=conversation_id,
            content=answer,
            metadata={
                **metadata,
                "sources": [source.model_dump() for source in sources],
                "citations": [citation.model_dump() for citation in citations],
            },
        )

        return ChatResponse(
            query=request.query,
            answer=answer,
            conversation_id=conversation_id,
            message_id=assistant_message.id if assistant_message else None,
            sources=sources,
            citations=citations,
            context_text=compression_result.context_text,
            prompt_text=prompt_result.prompt_text,
            llm_model=llm_response.model,
            metadata=metadata,
        )

    def _call_llm_with_tool_fallback(self, llm, request: LLMRequest):
        if not request.tools:
            return llm.chat(request)

        try:
            return llm.chat(request)
        except Exception:
            logger.exception("LLM tool calling failed, retry without tools")
            fallback_request = LLMRequest(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                metadata={
                    **request.metadata,
                    "tool_calling_fallback": True,
                },
            )
            response = llm.chat(fallback_request)
            response.metadata["tool_calling_fallback"] = True
            return response

    def _summarize_tool_results(
        self,
        llm,
        prompt_messages: list[LLMMessage],
        tool_results: list[dict],
    ) -> tuple[str, bool]:
        fallback_answer = self._build_tool_result_answer(tool_results)
        try:
            summary_response = llm.chat(
                LLMRequest(
                    messages=[
                        *prompt_messages,
                        LLMMessage(
                            role="user",
                            content=(
                                "以下是工具执行结果，请基于原问题和工具结果给出自然语言回答：\n"
                                f"{tool_results}"
                            ),
                        ),
                    ],
                    metadata={
                        "tools_enabled": True,
                        "tool_summary": True,
                    },
                )
            )
            return summary_response.answer or fallback_answer, True
        except Exception:
            logger.exception("LLM tool result summary failed")
            return fallback_answer, False

    def _to_function_tool(self, tool_definition: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool_definition["name"],
                "description": tool_definition["description"],
                "parameters": tool_definition["parameters"],
            },
        }

    def _prepare_conversation(self, request: ChatRequest) -> tuple[int | None, int]:
        if self.conversation_service is None:
            return request.conversation_id, 0

        if request.conversation_id is None:
            conversation = self.conversation_service.create_conversation(
                ConversationCreate(
                    title=self._build_conversation_title(request.query),
                    knowledge_base_id=request.knowledge_base_id,
                )
            )
        else:
            conversation = self.conversation_service.get_conversation(
                request.conversation_id
            )

        history = self.conversation_service.get_recent_messages(conversation.id)
        return conversation.id, len(history)

    def _save_assistant_message(
        self,
        conversation_id: int | None,
        content: str,
        metadata: dict,
    ) -> Message | None:
        if self.conversation_service is None or conversation_id is None:
            return None

        return self.conversation_service.add_assistant_message(
            conversation_id=conversation_id,
            content=content,
            metadata=metadata,
        )

    def _build_conversation_title(self, query: str) -> str:
        title = query.strip()[:20]
        return title or "New Chat"

    def _build_tool_result_answer(self, tool_results: list[dict]) -> str:
        if not tool_results:
            return ""

        lines = []
        for result in tool_results:
            if result.get("success"):
                lines.append(f"{result.get('name')}: {result.get('result')}")
            else:
                lines.append(f"{result.get('name')}: {result.get('error')}")
        return "\n".join(lines)
