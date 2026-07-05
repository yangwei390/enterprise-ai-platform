from fastapi import APIRouter

from backend.app.retrievers import RetrieverFactory, RetrieveQuery
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.retriever import RetrieveRequest, RetrieveResponse


router = APIRouter()


@router.post("/retriever/search", response_model=ApiResponse)
def search(request: RetrieveRequest) -> ApiResponse:
    retriever = RetrieverFactory.get_retriever()
    result = retriever.retrieve(
        RetrieveQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
        )
    )
    return success(data=RetrieveResponse.model_validate(result.model_dump()))
