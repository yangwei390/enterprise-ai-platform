from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieveQuery, RetrieverFactory
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.retriever import RetrieveRequest, RetrieveResponse
from fastapi import APIRouter

router = APIRouter()


@router.post("/retriever/search", response_model=ApiResponse)
def search(request: RetrieveRequest) -> ApiResponse:
    retriever = RetrieverFactory.get_retriever()
    retrieve_result = retriever.retrieve(
        RetrieveQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
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
    context_builder = ContextBuilderFactory.get_builder()
    context_result = context_builder.build(
        ContextBuildRequest(
            query=request.query,
            chunks=rerank_result.chunks,
        )
    )
    prompt_builder = PromptBuilderFactory.get_builder()
    prompt_result = prompt_builder.build(
        PromptBuildRequest(
            query=request.query,
            context_text=context_result.context_text,
        )
    )
    response_data = {
        **rerank_result.model_dump(),
        "context_text": context_result.context_text,
        "context_total_chars": context_result.total_chars,
        "context_chunks": [chunk.model_dump() for chunk in context_result.chunks],
        "prompt_text": prompt_result.prompt_text,
        "prompt_messages": [message.model_dump() for message in prompt_result.messages],
    }
    return success(data=RetrieveResponse.model_validate(response_data))
