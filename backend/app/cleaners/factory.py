from backend.app.cleaners.base import BaseCleaner
from backend.app.cleaners.basic_cleaner import BasicTextCleaner


class CleanerFactory:
    @staticmethod
    def get_cleaner(suffix: str) -> BaseCleaner:
        return BasicTextCleaner()
