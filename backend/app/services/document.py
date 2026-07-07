from __future__ import annotations

from pathlib import Path

from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.models import Document
from backend.app.pipeline import DocumentPipeline
from backend.app.repositories.document import DocumentRepository
from backend.app.schemas.document import DocumentCreate, DocumentUpdate
from backend.app.services.base import BaseService
from backend.app.storage import LocalStorageService
from fastapi import UploadFile


class DocumentService(BaseService[DocumentRepository]):
    def create(self, data: DocumentCreate) -> Document:
        logger.info(
            f"Create document metadata started | knowledge_base_id={data.knowledge_base_id}"
        )
        self._validate_create(data)
        self._validate_knowledge_base_exists(data.knowledge_base_id)

        document = self.repository.create(data.model_dump())
        logger.info("Create document metadata succeeded")
        return document

    def upload_document(self, knowledge_base_id: int | None, file: UploadFile) -> Document:
        logger.info(f"Upload document started | knowledge_base_id={knowledge_base_id}")
        if knowledge_base_id is None:
            logger.warning("Knowledge base id is empty")
            raise BusinessException(40002, "知识库ID不能为空")
        self._validate_knowledge_base_exists(knowledge_base_id)

        storage_service = LocalStorageService()
        storage_data = storage_service.save_upload_file(file)
        document_data = {
            "knowledge_base_id": knowledge_base_id,
            "filename": Path(storage_data["storage_path"]).name,
            "file_size": storage_data["file_size"],
            "status": "uploaded",
            "chunk_count": 0,
            "original_filename": storage_data["original_filename"],
            "storage_path": storage_data["storage_path"],
            "mime_type": storage_data["mime_type"],
            "file_hash": storage_data["file_hash"],
            "parse_status": "pending",
            "parse_message": None,
        }

        document = self.repository.create(document_data)
        logger.info("Upload document succeeded")
        return document

    def parse_document(self, document_id: int) -> dict:
        logger.info("Parse document started")
        document = self.repository.get(document_id)
        if document is None:
            logger.warning("Document not found")
            raise BusinessException(40402, "文档不存在")

        self.repository.update(
            document,
            {"parse_status": "processing", "parse_message": None},
        )

        try:
            pipeline = DocumentPipeline()
            context = pipeline.run(document)
        except BusinessException as exc:
            self.repository.update(
                document,
                {"parse_status": "failed", "parse_message": exc.message},
            )
            logger.warning("Parse document failed")
            raise
        except Exception as exc:
            self.repository.update(
                document,
                {"parse_status": "failed", "parse_message": str(exc)},
            )
            logger.exception("Parse document crashed")
            raise BusinessException(41003, "文档解析失败") from exc

        self.repository.update(
            document,
            {"parse_status": "success", "parse_message": "解析成功"},
        )
        parse_result = context.parse_result
        clean_result = context.clean_result
        chunk_result = context.chunk_result
        embedding_result = context.embedding_result
        vector_store_result = context.vector_store_result
        if (
            parse_result is None
            or clean_result is None
            or chunk_result is None
            or embedding_result is None
            or vector_store_result is None
        ):
            logger.warning("Document pipeline result is incomplete")
            raise BusinessException(41003, "文档解析失败")

        chunks_preview = [
            {
                "document_id": chunk.document_id,
                "knowledge_base_id": chunk.knowledge_base_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text[:200],
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            for chunk in chunk_result.chunks[:3]
        ]
        embeddings_preview = [
            {
                "chunk_index": item.chunk_index,
                "document_id": item.document_id,
                "knowledge_base_id": item.knowledge_base_id,
                "dimension": len(item.vector),
                "preview": item.vector[:5],
            }
            for item in embedding_result.items[:3]
        ]

        logger.info("Parse document succeeded")
        return {
            "document_id": document.id,
            "text_length": clean_result.cleaned_length,
            "preview": clean_result.text[:500],
            "page_count": parse_result.page_count,
            "metadata": {
                **parse_result.metadata,
                "bm25_indexed": context.metadata.get("bm25_indexed", False),
                "bm25_replaced": context.metadata.get("bm25_replaced", False),
                "bm25_indexed_count": context.metadata.get("bm25_indexed_count", 0),
                "bm25_index_path": context.metadata.get("bm25_index_path"),
                "bm25_error": context.metadata.get("bm25_error"),
            },
            "original_length": clean_result.original_length,
            "cleaned_length": clean_result.cleaned_length,
            "cleaner_metadata": clean_result.metadata,
            "chunk_strategy": chunk_result.strategy,
            "chunk_size": chunk_result.chunk_size,
            "chunk_overlap": chunk_result.chunk_overlap,
            "total_chunks": chunk_result.total_chunks,
            "chunks_preview": chunks_preview,
            "embedding_model": embedding_result.model_name,
            "embedding_dimension": embedding_result.dimension,
            "total_embeddings": embedding_result.total_items,
            "embeddings_preview": embeddings_preview,
            "vector_collection": vector_store_result.collection_name,
            "vector_total_records": vector_store_result.total_records,
            "vector_ids_preview": vector_store_result.ids[:5],
        }

    def get(self, id: int) -> Document:
        logger.info("Get document started")
        document = self.repository.get(id)
        if document is None:
            logger.warning("Document not found")
            raise BusinessException(40402, "文档不存在")

        logger.info("Get document succeeded")
        return document

    def list(self) -> list[Document]:
        logger.info("List documents started")
        documents = self.repository.list()
        logger.info("List documents succeeded")
        return documents

    def list_by_knowledge_base_id(self, knowledge_base_id: int) -> list[Document]:
        logger.info("List documents by knowledge base started")
        if knowledge_base_id is None:
            logger.warning("Knowledge base id is empty")
            raise BusinessException(40002, "知识库ID不能为空")

        documents = self.repository.list_by_knowledge_base_id(knowledge_base_id)
        logger.info("List documents by knowledge base succeeded")
        return documents

    def update(self, id: int, data: DocumentUpdate) -> Document:
        logger.info("Update document started")
        update_data = data.model_dump(exclude_unset=True)
        document = self.get(id)
        document = self.repository.update(document, update_data)
        logger.info("Update document succeeded")
        return document

    def delete(self, id: int) -> None:
        logger.info("Delete document started")
        document = self.get(id)
        self.repository.delete(document)
        logger.info("Delete document succeeded")

    def _validate_create(self, data: DocumentCreate) -> None:
        if data.knowledge_base_id is None:
            logger.warning("Knowledge base id is empty")
            raise BusinessException(40002, "知识库ID不能为空")

        if not data.filename or not data.filename.strip():
            logger.warning("Document filename is empty")
            raise BusinessException(40003, "文件名不能为空")

        if data.file_size < 0:
            logger.warning("Document file size is invalid")
            raise BusinessException(40004, "文件大小不能小于0")

    def _validate_knowledge_base_exists(self, knowledge_base_id: int | None) -> None:
        logger.info(f"Validate knowledge base exists | knowledge_base_id={knowledge_base_id}")
        if knowledge_base_id is None:
            logger.warning("Knowledge base id is empty")
            raise BusinessException(40002, "知识库ID不能为空")

        if not self.repository.knowledge_base_exists(knowledge_base_id):
            logger.warning(f"Knowledge base not found | knowledge_base_id={knowledge_base_id}")
            raise BusinessException(40401, "知识库不存在")
