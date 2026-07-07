from typing import Any

from backend.app.embeddings import EmbeddingFactory
from backend.app.embeddings.config import get_embedding_config
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.pipeline import DocumentPipeline
from backend.app.repositories.document import DocumentRepository
from backend.app.repositories.knowledge_base import KnowledgeBaseRepository
from backend.app.services.base import BaseService


class KnowledgeBaseReindexService(BaseService[KnowledgeBaseRepository]):
    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
    ) -> None:
        super().__init__(knowledge_base_repository)
        self.document_repository = document_repository

    def reindex(self, knowledge_base_id: int) -> dict[str, Any]:
        logger.info(f"Knowledge base reindex started | knowledge_base_id={knowledge_base_id}")
        knowledge_base = self.repository.get(knowledge_base_id)
        if knowledge_base is None:
            logger.warning("Knowledge base not found")
            raise BusinessException(40401, "知识库不存在")

        documents = self.document_repository.list_active_by_knowledge_base_id(
            knowledge_base_id
        )
        config = get_embedding_config()
        embedding = EmbeddingFactory.get_embedding()
        result: dict[str, Any] = {
            "knowledge_base_id": knowledge_base_id,
            "document_count": len(documents),
            "chunk_count": 0,
            "embedding_provider": config.provider,
            "embedding_model": embedding.model_name,
            "embedding_dimension": embedding.dimension,
            "vector_collection": None,
            "success_count": 0,
            "failed_count": 0,
            "errors": [],
        }

        pipeline = DocumentPipeline()
        for document in documents:
            self.document_repository.update(
                document,
                {"parse_status": "processing", "parse_message": None},
            )
            try:
                context = pipeline.run(document)
                if context.chunk_result is None or context.embedding_result is None:
                    raise BusinessException(41003, "文档解析失败")

                result["chunk_count"] += context.chunk_result.total_chunks
                result["embedding_provider"] = context.embedding_result.metadata.get(
                    "embedding_provider",
                    config.provider,
                )
                result["embedding_model"] = context.embedding_result.model_name
                result["embedding_dimension"] = context.embedding_result.dimension
                if context.vector_store_result is not None:
                    result["vector_collection"] = (
                        context.vector_store_result.collection_name
                    )

                self.document_repository.update(
                    document,
                    {
                        "parse_status": "success",
                        "parse_message": "重新索引成功",
                        "chunk_count": context.chunk_result.total_chunks,
                    },
                )
                result["success_count"] += 1
            except BusinessException as exc:
                self.document_repository.update(
                    document,
                    {"parse_status": "failed", "parse_message": exc.message},
                )
                result["failed_count"] += 1
                result["errors"].append(
                    {
                        "document_id": document.id,
                        "filename": document.original_filename or document.filename,
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
                logger.warning(
                    f"Knowledge base reindex document failed | "
                    f"document_id={document.id} | error={exc.message}"
                )
            except Exception as exc:
                error_message = str(exc)
                self.document_repository.update(
                    document,
                    {"parse_status": "failed", "parse_message": error_message},
                )
                result["failed_count"] += 1
                result["errors"].append(
                    {
                        "document_id": document.id,
                        "filename": document.original_filename or document.filename,
                        "code": 41003,
                        "message": error_message,
                    }
                )
                logger.exception("Knowledge base reindex document crashed")

        logger.info(
            f"Knowledge base reindex finished | knowledge_base_id={knowledge_base_id} | "
            f"success_count={result['success_count']} | failed_count={result['failed_count']}"
        )
        return result
