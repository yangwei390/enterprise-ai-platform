from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.repositories.knowledge_base import KnowledgeBaseRepository
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from backend.app.services.knowledge_base import KnowledgeBaseService


router = APIRouter()


def get_knowledge_base_service(db: Session = Depends(get_db)) -> KnowledgeBaseService:
    repository = KnowledgeBaseRepository(db)
    return KnowledgeBaseService(repository)


@router.get("/kb", response_model=ApiResponse)
def list_knowledge_bases(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> ApiResponse:
    knowledge_bases = service.list()
    items = [KnowledgeBaseResponse.model_validate(item) for item in knowledge_bases]
    return success(data=KnowledgeBaseListResponse(items=items, total=len(items)))


@router.get("/kb/{id}", response_model=ApiResponse)
def get_knowledge_base(
    id: int,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> ApiResponse:
    knowledge_base = service.get(id)
    return success(data=KnowledgeBaseResponse.model_validate(knowledge_base))


@router.post("/kb", response_model=ApiResponse)
def create_knowledge_base(
    data: KnowledgeBaseCreate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> ApiResponse:
    knowledge_base = service.create(data)
    return success(data=KnowledgeBaseResponse.model_validate(knowledge_base))


@router.put("/kb/{id}", response_model=ApiResponse)
def update_knowledge_base(
    id: int,
    data: KnowledgeBaseUpdate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> ApiResponse:
    knowledge_base = service.update(id, data)
    return success(data=KnowledgeBaseResponse.model_validate(knowledge_base))


@router.delete("/kb/{id}", response_model=ApiResponse)
def delete_knowledge_base(
    id: int,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> ApiResponse:
    service.delete(id)
    return success(data={"deleted": True})
