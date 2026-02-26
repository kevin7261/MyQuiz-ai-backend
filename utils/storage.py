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
# 上傳時未提供 person_id 時使用的目錄名
UPLOAD_DEFAULT_PERSON = "_"


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
    person_id: str | None = None,
    file_id: str | None = None,
) -> str:
    """
    將 ZIP 內容寫入後端儲存，回傳 file_id。
    folder 可為 FOLDER_UPLOAD（上傳）、FOLDER_REPACK（重新壓縮）、FOLDER_RAG（RAG 向量庫）。
    上傳時可傳入 file_id（由 API 指定）；未傳入或 repack/rag 時自動產生 UUID。
    上傳的 ZIP 存於 storage/{person_id}/{file_id}/upload/；無 person_id 時使用 storage/_/{file_id}/upload/。
    repack/rag 仍存於 storage/repack/、storage/rag/。
    其他 API 可用 get_zip_path(file_id) 取得檔案路徑後讀取。
    """
    if file_id is not None and ("/" in file_id or "\\" in file_id or not file_id.strip()):
        raise ValueError("file_id 不可包含路徑字元且不可為空")
    if folder == FOLDER_UPLOAD and file_id:
        file_id = file_id.strip()
    else:
        file_id = str(uuid.uuid4())
    if folder == FOLDER_UPLOAD:
        pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
        # 避免 path traversal：僅保留安全目錄名
        if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
            pid = UPLOAD_DEFAULT_PERSON
        target_dir = _storage_base() / pid / file_id / FOLDER_UPLOAD
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = _folder_dir(folder)
    path = target_dir / f"{file_id}.zip"
    path.write_bytes(contents)
    meta = _load_metadata()
    meta[file_id] = {
        "filename": original_filename or f"{file_id}.zip",
        "folder": folder,
    }
    if folder == FOLDER_UPLOAD:
        pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
        if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
            pid = UPLOAD_DEFAULT_PERSON
        meta[file_id]["person_id"] = pid
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
    上傳檔路徑為 storage/{person_id}/{file_id}/upload/{file_id}.zip；repack/rag 為 storage/repack/、storage/rag/。
    其他 API 可這樣使用：
        path = get_zip_path(file_id)
        if path and path.exists():
            with zipfile.ZipFile(path, "r") as z: ...
    """
    if not file_id or "/" in file_id or "\\" in file_id:
        return None
    meta = _load_metadata()
    entry = meta.get(file_id)
    folder = None
    person_id = None
    if entry is not None:
        if isinstance(entry, dict):
            folder = entry.get("folder", FOLDER_UPLOAD)
            person_id = entry.get("person_id")
        else:
            folder = FOLDER_UPLOAD
    if folder is not None:
        if folder == FOLDER_UPLOAD:
            pid = person_id or file_id  # 舊版 metadata 無 person_id，用 file_id 當目錄
            path = _storage_base() / pid / file_id / FOLDER_UPLOAD / f"{file_id}.zip"
        else:
            path = _folder_dir(folder) / f"{file_id}.zip"
        if path.exists():
            return path
    # 舊版：storage/{file_id}/upload/
    upload_by_file_id = _storage_base() / file_id / FOLDER_UPLOAD / f"{file_id}.zip"
    if upload_by_file_id.exists():
        return upload_by_file_id
    path = _folder_dir(FOLDER_UPLOAD) / f"{file_id}.zip"
    if path.exists():
        return path
    legacy = _storage_base() / f"{file_id}.zip"
    return legacy if legacy.exists() else None


def delete_zip(file_id: str) -> bool:
    """
    不再實際刪除：ZIP 永久保留。保留此 API 相容性，但不會刪除檔案或 metadata。
    """
    return False


def clear_folders(folders: list[str]) -> int:
    """
    不再實際刪除：ZIP 永久保留。保留此 API 相容性，但不刪除任何檔案。
    回傳 0。
    """
    return 0
