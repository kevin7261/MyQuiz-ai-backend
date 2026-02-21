"""
上傳檔案的儲存與解析。
ZIP 依 file_id 存成 storage/{file_id}.zip，其他 API 可用 get_zip_path(file_id) 取得路徑讀取。
"""

import os
import uuid
from pathlib import Path


def _storage_dir() -> Path:
    """儲存目錄，可由環境變數 ZIP_STORAGE_DIR 指定，預設為專案下的 storage/。"""
    base = os.environ.get("ZIP_STORAGE_DIR", "storage")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_zip(contents: bytes, original_filename: str | None = None) -> str:
    """
    將 ZIP 內容寫入後端儲存，回傳唯一 file_id。
    其他 API 可用 get_zip_path(file_id) 取得檔案路徑後讀取。
    """
    file_id = str(uuid.uuid4())
    path = _storage_dir() / f"{file_id}.zip"
    path.write_bytes(contents)
    return file_id


def get_zip_path(file_id: str) -> Path | None:
    """
    依 file_id 取得已儲存的 ZIP 檔案路徑；不存在則回傳 None。
    其他 API 可這樣使用：
        path = get_zip_path(file_id)
        if path and path.exists():
            with zipfile.ZipFile(path, "r") as z: ...
    """
    if not file_id or "/" in file_id or "\\" in file_id:
        return None
    path = _storage_dir() / f"{file_id}.zip"
    return path if path.exists() else None
