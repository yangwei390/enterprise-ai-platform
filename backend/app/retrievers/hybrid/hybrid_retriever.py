from backend.app.retrievers.hybrid.base import HybridRetrieveQuery, HybridRetrieveResult
from backend.app.retrievers.hybrid.dense_retriever import DenseRetriever
from backend.app.retrievers.hybrid.sparse_retriever import BM25SparseRetriever
from backend.app.retrievers.pipeline import RetrieverPipeline, RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps import (
    DenseRetrieveStep,
    FusionStep,
    MetadataFilterStep,
    QueryRewriteStep,
    SoftBoostStep,
    SparseRetrieveStep,
)


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
        pipeline: RetrieverPipeline | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()
        self.pipeline = pipeline or RetrieverPipeline(
            steps=[
                QueryRewriteStep(),
                MetadataFilterStep(),
                DenseRetrieveStep(self.dense_retriever),
                SparseRetrieveStep(self.sparse_retriever),
                FusionStep(),
                SoftBoostStep(),
            ]
        )

    def retrieve(self, query: HybridRetrieveQuery) -> HybridRetrieveResult:
        context = self.pipeline.run(
            RetrieverPipelineContext(
                query=query.query,
                knowledge_base_id=query.knowledge_base_id,
                top_k=query.top_k,
                score_threshold=query.score_threshold,
                metadata_filter=query.metadata_filter,
            )
        )

        return HybridRetrieveResult(
            chunks=context.fused_chunks,
            total=len(context.fused_chunks),
            metadata={
                "retriever_mode": "hybrid",
                "dense_total": len(context.dense_chunks),
                "sparse_total": len(context.sparse_chunks),
                "fused_total": len(context.fused_chunks),
                "bm25_enabled": True,
                **context.metadata,
                "errors": context.errors,
            },
        )
