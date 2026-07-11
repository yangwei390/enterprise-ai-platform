from pathlib import Path

from backend.app.chunkers import ChunkerFactory
from backend.app.chunkers.router import ChunkStrategyRouter
from backend.app.cleaners import CleanerFactory
from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.documents import DocumentClassifier, StructureParserFactory
from backend.app.embeddings import EmbeddingFactory
from backend.app.exceptions import BusinessException
from backend.app.indexing import DocumentIndexSynchronizer
from backend.app.parsers import ParserFactory
from backend.app.pipeline.base import PipelineContext, PipelineStep
from backend.app.vectorstores import VectorRecord, VectorStoreFactory


class ParserStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        storage_path = getattr(context.document, "storage_path", None)
        if not storage_path:
            raise BusinessException(41002, "文档文件不存在")

        upload_dir = Path(settings.UPLOAD_DIR)
        if not upload_dir.is_absolute():
            upload_dir = PROJECT_ROOT / upload_dir

        file_path = upload_dir / storage_path
        if not file_path.exists():
            raise BusinessException(41002, "文档文件不存在")

        parser = ParserFactory.get_parser(file_path)
        context.parse_result = parser.parse(file_path)
        context.metadata["file_path"] = file_path
        context.metadata["suffix"] = file_path.suffix.lower()
        context.metadata["parser"] = parser.__class__.__name__
        return context


class CleanerStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.parse_result is None:
            raise BusinessException(41003, "文档解析失败")

        suffix = context.metadata.get("suffix", "")
        cleaner = CleanerFactory.get_cleaner(suffix)
        context.clean_result = cleaner.clean(context.parse_result.text)
        context.metadata["cleaner"] = cleaner.__class__.__name__
        return context


class ChunkStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.clean_result is None:
            raise BusinessException(41003, "文档解析失败")

        suffix = context.metadata.get("suffix")
        document = context.document
        source = getattr(document, "original_filename", None) or getattr(document, "filename", None)
        base_metadata = {
            "document_id": getattr(document, "id", None),
            "knowledge_base_id": getattr(document, "knowledge_base_id", None),
            "source": source,
            "filename": getattr(document, "filename", None),
            "original_filename": getattr(document, "original_filename", None),
            "mime_type": getattr(document, "mime_type", None),
            "parser": context.metadata.get("parser"),
            "cleaner": context.metadata.get("cleaner"),
            "page_count": context.parse_result.page_count if context.parse_result else None,
            "suffix": suffix,
        }
        try:
            chunk_metadata = self._build_advanced_chunk_metadata(
                text=context.clean_result.text,
                base_metadata=base_metadata,
            )
            strategy = chunk_metadata["chunk_strategy_actual"]
            chunker = ChunkerFactory.get_chunker(suffix, strategy=strategy)
            context.chunk_result = chunker.chunk(
                context.clean_result.text,
                metadata=chunk_metadata,
            )
            self._clean_internal_metadata(context.chunk_result)
            context.metadata["document_structure"] = chunk_metadata.get("document_structure")
            context.metadata["chunking"] = {
                "requested_strategy": chunk_metadata.get("chunk_strategy_requested"),
                "actual_strategy": strategy,
                "chunk_count": context.chunk_result.total_chunks,
                "parent_count": sum(
                    1
                    for chunk in context.chunk_result.chunks
                    if chunk.metadata.get("chunk_role") == "parent"
                ),
                "child_count": sum(
                    1
                    for chunk in context.chunk_result.chunks
                    if chunk.metadata.get("chunk_role") == "child"
                ),
                "fallback_used": context.chunk_result.metadata.get("fallback_used", False),
                "fallback_reason": context.chunk_result.metadata.get("fallback_reason"),
            }
        except Exception as exc:
            fallback = ChunkerFactory.get_chunker(suffix, strategy="fixed")
            context.chunk_result = fallback.chunk(
                context.clean_result.text,
                metadata={**base_metadata, "document_type": "plain_text"},
            )
            context.chunk_result.metadata["fallback_used"] = True
            context.chunk_result.metadata["fallback_reason"] = str(exc)
            context.chunk_result.metadata["fallback_chain"] = [
                "advanced",
                "fixed",
            ]
            context.metadata["chunking"] = {
                "requested_strategy": settings.CHUNK_STRATEGY,
                "actual_strategy": "fixed",
                "chunk_count": context.chunk_result.total_chunks,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }
        return context

    def _build_advanced_chunk_metadata(self, *, text: str, base_metadata: dict) -> dict:
        if not settings.DOCUMENT_CLASSIFICATION_ENABLED:
            return {
                **base_metadata,
                "document_type": "plain_text",
                "chunk_strategy_requested": settings.CHUNK_STRATEGY,
                "chunk_strategy_actual": "fixed",
            }

        classification = DocumentClassifier().classify(
            text=text,
            filename=base_metadata.get("original_filename") or base_metadata.get("filename"),
            mime_type=base_metadata.get("mime_type"),
            metadata=base_metadata,
        )
        structure = None
        structure_metadata = {
            "document_type": classification.document_type,
            "parser": None,
            "node_count": 0,
            "max_depth": 0,
            "parse_failed": False,
        }
        if settings.DOCUMENT_STRUCTURE_ENABLED:
            try:
                structure = StructureParserFactory.parse(
                    text,
                    {**base_metadata, "document_type": classification.document_type},
                    classification.document_type,
                )
                structure_metadata = {
                    "document_type": structure.document_type,
                    "parser": StructureParserFactory.get_parser(
                        classification.document_type
                    ).__class__.__name__,
                    "node_count": structure.metadata.get("node_count", len(structure.nodes)),
                    "max_depth": structure.metadata.get("max_depth", 0),
                    "parse_failed": structure.metadata.get("parse_failed", False),
                }
            except Exception as exc:
                if not settings.DOCUMENT_STRUCTURE_FAIL_OPEN:
                    raise
                structure_metadata = {
                    **structure_metadata,
                    "parse_failed": True,
                    "error": str(exc),
                }
        decision = ChunkStrategyRouter().route(
            document_type=structure.document_type if structure else classification.document_type,
            structure=structure,
            metadata=base_metadata,
        )
        return {
            **base_metadata,
            "document_type": structure.document_type if structure else classification.document_type,
            "document_classification": classification.model_dump(),
            "document_structure": structure_metadata,
            "_document_structure": structure,
            **decision.to_metadata(),
        }

    def _clean_internal_metadata(self, chunk_result) -> None:
        chunk_result.metadata.pop("_document_structure", None)
        for chunk in chunk_result.chunks:
            chunk.metadata.pop("_document_structure", None)


class EmbeddingStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.chunk_result is None:
            raise BusinessException(41003, "文档解析失败")

        embedding = EmbeddingFactory.get_embedding()
        context.embedding_result = embedding.embed_chunks(context.chunk_result.chunks)
        return context


class VectorStoreStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.embedding_result is None:
            raise BusinessException(41003, "文档解析失败")

        chunks_by_index = {}
        if context.chunk_result is not None:
            chunks_by_index = {chunk.chunk_index: chunk for chunk in context.chunk_result.chunks}

        records = [
            VectorRecord(
                id=f"{item.document_id}_{item.chunk_index}",
                vector=item.vector,
                text=item.text,
                document_id=item.document_id,
                knowledge_base_id=item.knowledge_base_id,
                chunk_index=item.chunk_index,
                metadata={
                    **item.metadata,
                    "chunk_size": context.chunk_result.chunk_size if context.chunk_result else None,
                    "chunk_overlap": (
                        context.chunk_result.chunk_overlap if context.chunk_result else None
                    ),
                    "start_offset": (
                        chunks_by_index[item.chunk_index].start_offset
                        if item.chunk_index in chunks_by_index
                        else None
                    ),
                    "end_offset": (
                        chunks_by_index[item.chunk_index].end_offset
                        if item.chunk_index in chunks_by_index
                        else None
                    ),
                    "token_count": (
                        chunks_by_index[item.chunk_index].token_count
                        if item.chunk_index in chunks_by_index
                        else None
                    ),
                },
            )
            for item in context.embedding_result.items
        ]
        vector_store = VectorStoreFactory.get_vector_store()
        context.vector_store_result = vector_store.upsert(records)
        return context


class BM25IndexStep(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.chunk_result is None:
            context.metadata["bm25_indexed"] = False
            context.metadata["bm25_indexed_count"] = 0
            context.metadata["bm25_error"] = "chunk_result is empty"
            return context

        synchronizer = DocumentIndexSynchronizer()
        metadata = synchronizer.sync_bm25_for_document(
            document=context.document,
            chunks=context.chunk_result.chunks,
        )
        context.metadata.update(metadata)

        return context


class DocumentPipeline:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = [
            ParserStep(),
            CleanerStep(),
            ChunkStep(),
            EmbeddingStep(),
            VectorStoreStep(),
            BM25IndexStep(),
        ]

    def run(self, document) -> PipelineContext:
        context = PipelineContext(document=document)
        for step in self.steps:
            context = step.run(context)

        return context
