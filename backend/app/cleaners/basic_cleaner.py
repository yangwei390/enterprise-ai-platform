import re
import unicodedata

from backend.app.cleaners.base import BaseCleaner, CleanResult


class BasicTextCleaner(BaseCleaner):
    def clean(self, text: str) -> CleanResult:
        original_length = len(text)
        cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_text = self._remove_control_chars(cleaned_text)
        cleaned_text = re.sub(r"[ \t\f\v]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return CleanResult(
            text=cleaned_text,
            original_length=original_length,
            cleaned_length=len(cleaned_text),
            metadata={
                "normalized_newlines": True,
                "compressed_blank_lines": True,
                "compressed_spaces": True,
                "removed_control_chars": True,
            },
        )

    def _remove_control_chars(self, text: str) -> str:
        return "".join(
            char for char in text if char in {"\n", "\t"} or unicodedata.category(char) != "Cc"
        )
