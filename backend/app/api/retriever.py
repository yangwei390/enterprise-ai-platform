from backend.app.context import ContextBuilderFactory, ContextBuildRequest
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.prompts import PromptBuilderFactory, PromptBuildRequest
from backend.app.rerankers import RerankerFactory, RerankQuery
from backend.app.retrievers import RetrieveQuery, RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery
from backend.app.retrievers.sparse import (
    BM25Retriever,
    SparseDocument,
    SparseSearchQuery,
    get_bm25_index_manager,
)
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.retriever import RetrieveRequest, RetrieveResponse
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class BM25TestDocument(BaseModel):
    id: str
    text: str
    document_id: int | None = None
    knowledge_base_id: int | None = None
    chunk_index: int | None = None
    metadata: dict = Field(default_factory=dict)


class BM25TestRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5
    metadata_filter: dict | None = None
    documents: list[BM25TestDocument]


@router.post("/retriever/search", response_model=ApiResponse)
def search(request: RetrieveRequest) -> ApiResponse:
    retriever = RetrieverFactory.get_retriever()
    retrieve_result = retriever.retrieve(
        RetrieveQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            metadata_filter=request.metadata_filter,
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
    llm = LLMFactory.get_llm()
    llm_response = llm.chat(
        LLMRequest(
            messages=[
                LLMMessage(role=message.role, content=message.content)
                for message in prompt_result.messages
            ]
        )
    )
    response_data = {
        **rerank_result.model_dump(),
        "context_text": context_result.context_text,
        "context_total_chars": context_result.total_chars,
        "context_chunks": [chunk.model_dump() for chunk in context_result.chunks],
        "prompt_text": prompt_result.prompt_text,
        "prompt_messages": [message.model_dump() for message in prompt_result.messages],
        "answer": llm_response.answer,
        "llm_model": llm_response.model,
        "llm_usage": llm_response.usage,
        "metadata": {
            **retrieve_result.metadata,
            "reranker": rerank_result.metadata,
        },
    }
    return success(data=RetrieveResponse.model_validate(response_data))


@router.post("/retriever/hybrid-search", response_model=ApiResponse)
def hybrid_search(request: RetrieveRequest) -> ApiResponse:
    retriever = RetrieverFactory.get_hybrid_retriever()
    result = retriever.retrieve(
        HybridRetrieveQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            metadata_filter=request.metadata_filter,
        )
    )
    return success(
        data={
            "query": request.query,
            "top_k": request.top_k,
            "total": result.total,
            "chunks": [chunk.model_dump() for chunk in result.chunks],
            "metadata": result.metadata,
        }
    )


@router.post("/retriever/bm25-test", response_model=ApiResponse)
def bm25_test(request: BM25TestRequest) -> ApiResponse:
    retriever = BM25Retriever()
    retriever.add_documents(
        [
            SparseDocument(
                id=document.id,
                text=document.text,
                document_id=document.document_id,
                knowledge_base_id=document.knowledge_base_id,
                chunk_index=document.chunk_index,
                metadata=document.metadata,
            )
            for document in request.documents
        ]
    )
    results = retriever.retrieve(
        SparseSearchQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )
    )
    return success(
        data={
            "query": request.query,
            "top_k": request.top_k,
            "total": len(results),
            "results": [result.model_dump() for result in results],
            "metadata": {
                "retriever": "bm25",
                "index": "memory",
                "documents_total": len(request.documents),
            },
        }
    )


@router.post("/retriever/bm25-manager-test", response_model=ApiResponse)
def bm25_manager_test(request: BM25TestRequest) -> ApiResponse:
    manager = get_bm25_index_manager()
    documents = [
        SparseDocument(
            id=document.id,
            text=document.text,
            document_id=document.document_id,
            knowledge_base_id=document.knowledge_base_id,
            chunk_index=document.chunk_index,
            metadata=document.metadata,
        )
        for document in request.documents
    ]

    manager.clear()
    manager.add_documents(documents)
    index = manager.get_index()
    results = index.search(
        SparseSearchQuery(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )
    )

    return success(
        data={
            "query": request.query,
            "top_k": request.top_k,
            "total": len(results),
            "results": [result.model_dump() for result in results],
            "metadata": {
                "index_path": str(manager.index_path),
                "total_docs": index.total_docs,
                "avg_doc_length": index.avg_doc_length,
            },
        }
    )
