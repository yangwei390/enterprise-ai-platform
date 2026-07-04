from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.models import Document
from backend.app.parsers import ParserFactory
from backend.app.repositories.document import DocumentRepository
from backend.app.schemas.document import DocumentCreate, DocumentUpdate
from backend.app.services.base import BaseService
from backend.app.storage import LocalStorageService


class DocumentService(BaseService[DocumentRepository]):
    def create(self, data: DocumentCreate) -> Document:
        logger.info("Create document metadata started")
        self._validate_create(data)

        document = self.repository.create(data.model_dump())
        logger.info("Create document metadata succeeded")
        return document

    def upload_document(self, knowledge_base_id: int | None, file: UploadFile) -> Document:
        logger.info("Upload document started")
        if knowledge_base_id is None:
            logger.warning("Knowledge base id is empty")
            raise BusinessException(40002, "知识库ID不能为空")

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

        if not document.storage_path:
            logger.warning("Document storage path is empty")
            raise BusinessException(41002, "文档文件不存在")

        upload_dir = Path(settings.UPLOAD_DIR)
        if not upload_dir.is_absolute():
            upload_dir = PROJECT_ROOT / upload_dir

        file_path = upload_dir / document.storage_path
        if not file_path.exists():
            logger.warning("Document file does not exist")
            raise BusinessException(41002, "文档文件不存在")

        self.repository.update(
            document,
            {"parse_status": "processing", "parse_message": None},
        )

        try:
            parser = ParserFactory.get_parser(file_path)
            parse_result = parser.parse(file_path)
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
        logger.info("Parse document succeeded")
        return {
            "document_id": document.id,
            "text_length": len(parse_result.text),
            "preview": parse_result.text[:500],
            "page_count": parse_result.page_count,
            "metadata": parse_result.metadata,
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
