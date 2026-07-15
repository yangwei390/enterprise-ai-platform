from pathlib import Path

from backend.app.exceptions import BusinessException
from backend.app.parsers.base import BaseParser
from backend.app.parsers.docx_parser import DocxParser
from backend.app.parsers.pdf_parser import PdfParser
from backend.app.parsers.text_parser import TextParser


class ParserFactory:
    @staticmethod
    def get_parser(file_path: Path) -> BaseParser:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return TextParser()

        if suffix == ".pdf":
            return PdfParser()

        if suffix == ".docx":
            return DocxParser()

        raise BusinessException(41001, "暂不支持该文件格式")
