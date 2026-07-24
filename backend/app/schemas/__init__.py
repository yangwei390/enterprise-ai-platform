from backend.app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
)
from backend.app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from backend.app.schemas.product import (
    ProductCreate,
    ProductDocumentLinkListResponse,
    ProductDocumentLinkResponse,
    ProductListResponse,
    ProductQuery,
    ProductRecommendationResponse,
    ProductResponse,
    ProductUpdate,
)
from backend.app.schemas.response import ApiResponse, error, success

__all__ = [
    "ApiResponse",
    "DocumentCreate",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentUpdate",
    "KnowledgeBaseCreate",
    "KnowledgeBaseListResponse",
    "KnowledgeBaseResponse",
    "KnowledgeBaseUpdate",
    "ProductCreate",
    "ProductDocumentLinkListResponse",
    "ProductDocumentLinkResponse",
    "ProductListResponse",
    "ProductQuery",
    "ProductRecommendationResponse",
    "ProductResponse",
    "ProductUpdate",
    "error",
    "success",
]
