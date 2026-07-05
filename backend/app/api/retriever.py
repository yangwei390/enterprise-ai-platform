from fastapi import APIRouter

from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieverFactory, RetrieveQuery
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.retriever import RetrieveRequest, RetrieveResponse


router = APIRouter()


@router.post("/retriever/search", response_model=ApiResponse)
def search(request: RetrieveRequest) -> ApiResponse:
    retriever = RetrieverFactory.get_retriever()
    retrieve_result = retriever.retrieve(
        RetrieveQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
        )
    )
    reranker = RerankerFactory.get_reranker()
    rerank_result = reranker.rerank(
        RerankQuery(
            query=request.query,
            chunks=retrieve_result.chunks,
            top_k=request.top_k,
        )
    )
    return success(data=RetrieveResponse.model_validate(rerank_result.model_dump()))
