from pathlib import Path

from backend.app.chunkers import ChunkerFactory
from backend.app.cleaners import CleanerFactory
from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.embeddings import EmbeddingFactory
from backend.app.exceptions import BusinessException
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
        chunker = ChunkerFactory.get_chunker(suffix)
        document = context.document
        context.chunk_result = chunker.chunk(
            context.clean_result.text,
            metadata={
                "document_id": getattr(document, "id", None),
                "knowledge_base_id": getattr(document, "knowledge_base_id", None),
                "source": getattr(document, "original_filename", None)
                or getattr(document, "filename", None),
                "parser": context.metadata.get("parser"),
                "cleaner": context.metadata.get("cleaner"),
                "page_count": context.parse_result.page_count if context.parse_result else None,
            },
        )
        return context


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


class DocumentPipeline:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = [
            ParserStep(),
            CleanerStep(),
            ChunkStep(),
            EmbeddingStep(),
            VectorStoreStep(),
        ]

    def run(self, document) -> PipelineContext:
        context = PipelineContext(document=document)
        for step in self.steps:
            context = step.run(context)

        return context
