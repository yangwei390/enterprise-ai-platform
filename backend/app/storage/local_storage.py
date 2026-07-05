import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config.settings import PROJECT_ROOT, settings
from fastapi import UploadFile


class LocalStorageService:
    def __init__(self) -> None:
        upload_dir = Path(settings.UPLOAD_DIR)
        if not upload_dir.is_absolute():
            upload_dir = PROJECT_ROOT / upload_dir

        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_upload_file(self, file: UploadFile) -> dict[str, Any]:
        original_filename = file.filename or "uploaded_file"
        suffix = Path(original_filename).suffix
        stored_filename = f"{uuid4().hex}{suffix}"
        date_dir = datetime.now().strftime("%Y/%m")
        relative_path = Path(date_dir) / stored_filename
        storage_path = self.upload_dir / relative_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = 0
        file_hash = hashlib.sha256()

        with storage_path.open("wb") as target_file:
            while chunk := file.file.read(1024 * 1024):
                file_size += len(chunk)
                file_hash.update(chunk)
                target_file.write(chunk)

        return {
            "original_filename": original_filename,
            "storage_path": relative_path.as_posix(),
            "file_size": file_size,
            "mime_type": file.content_type,
            "file_hash": file_hash.hexdigest(),
        }
