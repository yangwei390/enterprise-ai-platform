from pathlib import Path

from backend.app.chunkers import ChunkerFactory
from backend.app.cleaners import CleanerFactory
from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.exceptions import BusinessException
from backend.app.parsers import ParserFactory
from backend.app.pipeline.base import PipelineContext, PipelineStep


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
                "page_count": context.parse_result.page_count
                if context.parse_result
                else None,
            },
        )
        return context


class DocumentPipeline:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = [
            ParserStep(),
            CleanerStep(),
            ChunkStep(),
        ]

    def run(self, document) -> PipelineContext:
        context = PipelineContext(document=document)
        for step in self.steps:
            context = step.run(context)

        return context
