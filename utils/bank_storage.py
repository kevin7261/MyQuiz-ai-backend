"""
Bank 專屬上傳檔案儲存（自 utils.zip_storage 複製，與 rag 無關）。

所有 ZIP 存於 Supabase Storage bucket（環境變數 SUPABASE_BANK_BUCKET；未設定則沿用 SUPABASE_RAG_BUCKET，預設 "myQUIZ.ai"），
並一律加上 `bank/` 路徑前綴與獨立的 `_bank_metadata.json`，與 rag 的物件完全分開、不會衝突。

Bucket 內路徑結構：
  upload: bank/{person_id}/{page_id}/upload/{page_id}.zip
  repack: bank/{person_id}/{parent_page_id}/repack/{page_id}.zip
  rag:    bank/{person_id}/{parent_page_id}/rag/{page_id}.zip
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.taipei_time import TAIPEI_TZ

_logger = logging.getLogger(__name__)


FOLDER_UPLOAD = "upload"
FOLDER_REPACK = "repack"
FOLDER_RAG = "rag"
UPLOAD_DEFAULT_PERSON = "_"

# bank 專屬命名空間：路徑前綴與 metadata 物件，皆與 rag 分開
_BANK_PREFIX = "bank"
_METADATA_KEY = "_bank_metadata.json"


def _storage_safe_page_id(candidate: str, person_id: str | None = None) -> str:
    c = (candidate or "").strip()
    if not c or "/" in c or "\\" in c or len(c) > 255:
        return generate_page_id(person_id)
    if any(ord(ch) > 127 for ch in c):
        return generate_page_id(person_id)
    return c


def generate_page_id(person_id: str | None = None) -> str:
    safe = (person_id or "").strip() or "_"
    if "/" in safe or "\\" in safe:
        safe = "_"
    time_part = datetime.now(TAIPEI_TZ).strftime("%y%m%d%H%M%S")
    return f"{safe}_{time_part}"


def _get_bucket_name() -> str:
    return os.environ.get("SUPABASE_BANK_BUCKET") or os.environ.get("SUPABASE_RAG_BUCKET", "myQUIZ.ai")


def _get_storage():
    from utils.supabase import get_supabase
    return get_supabase().storage.from_(_get_bucket_name())


def _resolve_person_id(person_id: str | None) -> str:
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        return UPLOAD_DEFAULT_PERSON
    return pid


def _upload_object_basename(page_id: str) -> str:
    return f"{page_id}.zip"


def _is_storage_not_found(e: Exception) -> bool:
    code = getattr(e, "code", None)
    status = getattr(e, "status", None)
    return code == "not_found" or str(status) == "404"


def _load_metadata() -> dict:
    try:
        data = _get_storage().download(_METADATA_KEY)
    except Exception as e:
        if _is_storage_not_found(e):
            return {}
        raise
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        _logger.warning("_bank_metadata.json 內容損毀，視為空 metadata（將於下次寫入時重建）")
        return {}


def _save_metadata(data: dict) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage = _get_storage()
    try:
        storage.update(_METADATA_KEY, content, {"content-type": "application/json"})
    except Exception as e:
        if not _is_storage_not_found(e):
            raise
        storage.upload(_METADATA_KEY, content, {"content-type": "application/json"})


def _upload_to_bucket(storage_path: str, content: bytes) -> None:
    storage = _get_storage()
    try:
        storage.update(storage_path, content, {"content-type": "application/zip"})
    except Exception as e:
        if not _is_storage_not_found(e):
            raise
        storage.upload(storage_path, content, {"content-type": "application/zip"})


def save_zip(
    contents: bytes,
    original_filename: str | None = None,
    folder: str = FOLDER_UPLOAD,
    person_id: str | None = None,
    page_id: str | None = None,
    parent_page_id: str | None = None,
) -> str:
    """將 ZIP 上傳至 Supabase Storage（bank/ 命名空間），回傳 page_id。folder 可為 upload／repack／rag。"""
    if folder == FOLDER_UPLOAD:
        if page_id is not None and ("/" in page_id or "\\" in page_id or not page_id.strip()):
            raise ValueError("page_id 不可包含路徑字元且不可為空")
        page_id = (page_id or "").strip() or generate_page_id(person_id)
    else:
        if not parent_page_id or "/" in parent_page_id or "\\" in parent_page_id:
            raise ValueError("repack/rag 需傳入 parent_page_id（上傳的 page_id）")
        parent_page_id = parent_page_id.strip()
        raw = (page_id or "").strip()
        if raw and "/" not in raw and "\\" not in raw:
            page_id = _storage_safe_page_id(raw, person_id)
        else:
            page_id = generate_page_id(person_id)

    pid = _resolve_person_id(person_id)

    if folder == FOLDER_UPLOAD:
        display_name = (Path(original_filename).name if original_filename else "").strip() or _upload_object_basename(page_id)
        stored_name = _upload_object_basename(page_id)
        storage_path = f"{_BANK_PREFIX}/{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}"
    else:
        stored_name = f"{page_id}.zip"
        storage_path = f"{_BANK_PREFIX}/{pid}/{parent_page_id}/{folder}/{page_id}.zip"

    _upload_to_bucket(storage_path, contents)

    meta = _load_metadata()
    meta[page_id] = {
        "filename": display_name if folder == FOLDER_UPLOAD else (original_filename or stored_name),
        "folder": folder,
        "person_id": pid,
        "storage_path": storage_path,
    }
    if folder == FOLDER_UPLOAD:
        meta[page_id]["stored_filename"] = stored_name
    else:
        meta[page_id]["parent_page_id"] = parent_page_id
    _save_metadata(meta)
    return page_id


def _download_to_temp(storage_path: str) -> Optional[Path]:
    try:
        data = _get_storage().download(storage_path)
    except Exception as e:
        if _is_storage_not_found(e):
            return None
        raise
    if not data:
        return None
    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="myquizai_bank_dl_")
    os.close(fd)
    Path(tmp_path).write_bytes(data)
    return Path(tmp_path)


def get_zip_path(page_id: str) -> Optional[Path]:
    """依 page_id 從 Storage 下載 ZIP 至暫存檔後回傳路徑；不存在則回 None。呼叫端負責刪暫存檔。"""
    if not page_id or "/" in page_id or "\\" in page_id:
        return None

    meta = _load_metadata()
    entry = meta.get(page_id)

    if entry and isinstance(entry, dict):
        storage_path = entry.get("storage_path")
        if storage_path:
            return _download_to_temp(storage_path)

        folder = entry.get("folder", FOLDER_UPLOAD)
        pid = _resolve_person_id(entry.get("person_id"))
        parent_page_id = entry.get("parent_page_id")

        if folder == FOLDER_UPLOAD:
            stored_name = entry.get("stored_filename") or entry.get("filename") or f"{page_id}.zip"
            stored_name = Path(stored_name).name
            storage_path = f"{_BANK_PREFIX}/{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}"
        elif parent_page_id:
            storage_path = f"{_BANK_PREFIX}/{pid}/{parent_page_id}/{folder}/{page_id}.zip"
        else:
            return None

        return _download_to_temp(storage_path)

    return None


def get_zip_path_by_person(person_id: str, page_id: str) -> Optional[Path]:
    """依 person_id + page_id 下載上傳 ZIP 至暫存檔後回傳路徑。呼叫端負責刪暫存檔。"""
    if not page_id or "/" in page_id or "\\" in page_id:
        return None
    pid = _resolve_person_id(person_id)

    meta = _load_metadata()
    entry = meta.get(page_id)

    if entry and isinstance(entry, dict):
        storage_path = entry.get("storage_path")
        if storage_path:
            return _download_to_temp(storage_path)
        stored_name = entry.get("stored_filename") or entry.get("filename") or f"{page_id}.zip"
        stored_name = Path(stored_name).name
        return _download_to_temp(f"{_BANK_PREFIX}/{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}")

    return _download_to_temp(f"{_BANK_PREFIX}/{pid}/{page_id}/{FOLDER_UPLOAD}/{page_id}.zip")


def delete_tab_folder(person_id: str, page_id: str) -> bool:
    """刪除該 page_id 下 upload/repack/rag 所有檔案、以此為 parent 的子項目，並清除 metadata。回傳是否有刪除。"""
    if not page_id or "/" in page_id or "\\" in page_id:
        return False
    pid = _resolve_person_id(person_id)

    storage = _get_storage()
    deleted_something = False

    for folder in [FOLDER_UPLOAD, FOLDER_REPACK, FOLDER_RAG]:
        try:
            files = storage.list(f"{_BANK_PREFIX}/{pid}/{page_id}/{folder}")
            paths = [f"{_BANK_PREFIX}/{pid}/{page_id}/{folder}/{f['name']}" for f in (files or []) if f.get("name")]
            if paths:
                storage.remove(paths)
                deleted_something = True
        except Exception:
            pass

    meta = _load_metadata()
    child_ids = [k for k, v in meta.items() if isinstance(v, dict) and v.get("parent_page_id") == page_id]
    for child_id in child_ids:
        child_path = meta[child_id].get("storage_path")
        if child_path:
            try:
                storage.remove([child_path])
                deleted_something = True
            except Exception:
                pass

    to_remove = [k for k, v in meta.items() if k == page_id or (isinstance(v, dict) and v.get("parent_page_id") == page_id)]
    if to_remove:
        for k in to_remove:
            meta.pop(k, None)
        _save_metadata(meta)
        deleted_something = True

    return deleted_something
