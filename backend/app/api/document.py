from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.repositories.document import DocumentRepository
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentParseResponse,
    DocumentResponse,
    DocumentUpdate,
)
from backend.app.services.document import DocumentService


router = APIRouter()


def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    repository = DocumentRepository(db)
    return DocumentService(repository)


@router.get("/documents", response_model=ApiResponse)
def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    documents = service.list()
    items = [DocumentResponse.model_validate(item) for item in documents]
    return success(data=DocumentListResponse(items=items, total=len(items)))


@router.get("/documents/{id}", response_model=ApiResponse)
def get_document(
    id: int,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    document = service.get(id)
    return success(data=DocumentResponse.model_validate(document))


@router.get("/kb/{knowledge_base_id}/documents", response_model=ApiResponse)
def list_documents_by_knowledge_base(
    knowledge_base_id: int,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    documents = service.list_by_knowledge_base_id(knowledge_base_id)
    items = [DocumentResponse.model_validate(item) for item in documents]
    return success(data=DocumentListResponse(items=items, total=len(items)))


@router.post("/documents", response_model=ApiResponse)
def create_document(
    data: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    document = service.create(data)
    return success(data=DocumentResponse.model_validate(document))


@router.post("/documents/upload", response_model=ApiResponse)
def upload_document(
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    document = service.upload_document(knowledge_base_id, file)
    return success(data=DocumentResponse.model_validate(document))


@router.post("/documents/{id}/parse", response_model=ApiResponse)
def parse_document(
    id: int,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    result = service.parse_document(id)
    return success(data=DocumentParseResponse(**result))


@router.put("/documents/{id}", response_model=ApiResponse)
def update_document(
    id: int,
    data: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    document = service.update(id, data)
    return success(data=DocumentResponse.model_validate(document))


@router.delete("/documents/{id}", response_model=ApiResponse)
def delete_document(
    id: int,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse:
    service.delete(id)
    return success(data={"deleted": True})
