from backend.app.indexing import IndexVersionManager
from backend.app.rag import RagChatInput, RagChatPipeline
from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.schemas import KnowledgeSearchArgs, KnowledgeSearchOutput


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "基于企业知识库进行检索增强问答，返回答案、来源、引用和检索元数据。"
    args_schema = KnowledgeSearchArgs

    def run(self, arguments: dict) -> ToolResult:
        args = KnowledgeSearchArgs.model_validate(arguments)
        index_version = IndexVersionManager().get_version(args.knowledge_base_id)

        try:
            rag_result = RagChatPipeline().run(
                RagChatInput(
                    query=args.query,
                    knowledge_base_id=args.knowledge_base_id,
                    conversation_id=args.conversation_id,
                    memory_context=args.memory_context,
                )
            )
            output = KnowledgeSearchOutput(
                answer=rag_result.answer,
                sources=[source.model_dump() for source in rag_result.sources],
                citations=[citation.model_dump() for citation in rag_result.citations],
                metadata=rag_result.metadata,
            )
            return ToolResult(
                name=self.name,
                success=True,
                result=output.model_dump(),
                metadata={
                    "knowledge_base_id": args.knowledge_base_id,
                    "knowledge_base_index_version": index_version,
                    "conversation_id": args.conversation_id,
                    "failed": False,
                },
            )
        except Exception as exc:
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                metadata={
                    "knowledge_base_id": args.knowledge_base_id,
                    "knowledge_base_index_version": index_version,
                    "conversation_id": args.conversation_id,
                    "failed": True,
                    "error": str(exc),
                },
            )
