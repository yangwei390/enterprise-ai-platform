from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.models import KnowledgeBase
from backend.app.repositories.knowledge_base import KnowledgeBaseRepository
from backend.app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseUpdate
from backend.app.services.base import BaseService


class KnowledgeBaseService(BaseService[KnowledgeBaseRepository]):
    def create(self, data: KnowledgeBaseCreate) -> KnowledgeBase:
        logger.info("Create knowledge base started")
        self._validate_name(data.name)

        knowledge_base = self.repository.create(data.model_dump())
        logger.info("Create knowledge base succeeded")
        return knowledge_base

    def get(self, id: int) -> KnowledgeBase:
        logger.info("Get knowledge base started")
        knowledge_base = self.repository.get(id)
        if knowledge_base is None:
            logger.warning("Knowledge base not found")
            raise BusinessException(40401, "知识库不存在")

        logger.info("Get knowledge base succeeded")
        return knowledge_base

    def list(self) -> list[KnowledgeBase]:
        logger.info("List knowledge bases started")
        knowledge_bases = self.repository.list()
        logger.info("List knowledge bases succeeded")
        return knowledge_bases

    def update(self, id: int, data: KnowledgeBaseUpdate) -> KnowledgeBase:
        logger.info("Update knowledge base started")
        update_data = data.model_dump(exclude_unset=True)
        if "name" in update_data:
            self._validate_name(update_data["name"])

        knowledge_base = self.get(id)
        knowledge_base = self.repository.update(knowledge_base, update_data)
        logger.info("Update knowledge base succeeded")
        return knowledge_base

    def delete(self, id: int) -> None:
        logger.info("Delete knowledge base started")
        knowledge_base = self.get(id)
        self.repository.delete(knowledge_base)
        logger.info("Delete knowledge base succeeded")

    def _validate_name(self, name: str | None) -> None:
        if name is None or not name.strip():
            logger.warning("Knowledge base name is empty")
            raise BusinessException(40001, "知识库名称不能为空")
