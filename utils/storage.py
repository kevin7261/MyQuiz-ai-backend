"""
上傳檔案的儲存與解析。
ZIP 依類型存於不同子目錄：
- upload/：使用者上傳的 ZIP
- repack/：依資料夾重新壓縮的 ZIP
- rag/：RAG（FAISS）向量庫 ZIP
其他 API 可用 get_zip_path(file_id) 取得路徑讀取。
"""

import json
import os
import uuid
from pathlib import Path

# 子目錄名稱：上傳、重新壓縮、RAG
FOLDER_UPLOAD = "upload"
FOLDER_REPACK = "repack"
FOLDER_RAG = "rag"


def _storage_base() -> Path:
    """儲存根目錄，可由環境變數 ZIP_STORAGE_DIR 指定，預設為專案下的 storage/。"""
    base = os.environ.get("ZIP_STORAGE_DIR", "storage")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _folder_dir(folder: str) -> Path:
    """取得指定類型子目錄（upload / repack / rag），不存在會建立。"""
    path = _storage_base() / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path() -> Path:
    return _storage_base() / "_metadata.json"


def _load_metadata() -> dict:
    p = _metadata_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_metadata(data: dict) -> None:
    _metadata_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_zip(
    contents: bytes,
    original_filename: str | None = None,
    folder: str = FOLDER_UPLOAD,
) -> str:
    """
    將 ZIP 內容寫入後端儲存，回傳唯一 file_id。
    folder 可為 FOLDER_UPLOAD（上傳）、FOLDER_REPACK（重新壓縮）、FOLDER_RAG（RAG 向量庫）。
    其他 API 可用 get_zip_path(file_id) 取得檔案路徑後讀取。
    """
    file_id = str(uuid.uuid4())
    target_dir = _folder_dir(folder)
    path = target_dir / f"{file_id}.zip"
    path.write_bytes(contents)
    meta = _load_metadata()
    meta[file_id] = {
        "filename": original_filename or f"{file_id}.zip",
        "folder": folder,
    }
    _save_metadata(meta)
    return file_id


def get_zip_filename(file_id: str) -> str | None:
    """依 file_id 取得儲存時使用的檔名，供下載時 Content-Disposition 使用。"""
    meta = _load_metadata()
    entry = meta.get(file_id)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("filename")
    return entry


def _get_folder_for_file_id(file_id: str) -> str | None:
    """從 metadata 取得該 file_id 所屬子目錄；舊資料僅有 filename 字串時視為 upload。"""
    meta = _load_metadata()
    entry = meta.get(file_id)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("folder", FOLDER_UPLOAD)
    return FOLDER_UPLOAD


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
    folder = _get_folder_for_file_id(file_id)
    if folder is not None:
        path = _folder_dir(folder) / f"{file_id}.zip"
        if path.exists():
            return path
    # 舊版：metadata 無此 file_id 或檔案在根目錄
    for candidate in (FOLDER_UPLOAD,):  # 先找 upload
        path = _folder_dir(candidate) / f"{file_id}.zip"
        if path.exists():
            return path
    legacy = _storage_base() / f"{file_id}.zip"
    return legacy if legacy.exists() else None


def delete_zip(file_id: str) -> bool:
    """
    刪除指定 file_id 的 ZIP 檔與 metadata 紀錄。
    回傳是否成功刪除（有找到檔案且已刪除為 True）。
    """
    if not file_id or "/" in file_id or "\\" in file_id:
        return False
    path = get_zip_path(file_id)
    existed = path is not None and path.exists()
    if existed:
        try:
            path.unlink()
        except OSError:
            return False
    meta = _load_metadata()
    if file_id in meta:
        del meta[file_id]
        _save_metadata(meta)
    return existed


def clear_folders(folders: list[str]) -> int:
    """
    清空指定子目錄內所有 ZIP 檔並自 metadata 移除紀錄。
    例如 clear_folders([FOLDER_UPLOAD, FOLDER_REPACK, FOLDER_RAG]) 會刪除三個目錄內全部檔案。
    回傳刪除的檔案數量。
    """
    meta = _load_metadata()
    removed = 0
    meta_changed = False
    for folder in folders:
        target_dir = _folder_dir(folder)
        for path in target_dir.glob("*.zip"):
            file_id = path.stem
            if file_id in meta:
                del meta[file_id]
                meta_changed = True
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    if meta_changed:
        _save_metadata(meta)
    return removed
