"""
上傳檔案的儲存與解析模組（Supabase Storage 版本）。
所有 ZIP 存於 Supabase Storage bucket（環境變數 SUPABASE_RAG_BUCKET，預設 "MyQuiz-ai"）。

Bucket 內路徑結構：
  upload: {person_id}/{page_id}/upload/{page_id}.zip（物件名僅 ASCII；原始檔名存在 metadata.filename）
  repack: {person_id}/{parent_page_id}/repack/{page_id}.zip
  rag:    {person_id}/{parent_page_id}/rag/{page_id}.zip

Metadata 以 _metadata.json 存於 bucket 根目錄，供 get_zip_path 依 page_id 查路徑。

get_zip_path() / get_zip_path_by_person() 會將 ZIP 下載至暫存檔後回傳 Path；
  呼叫端負責用完後刪除暫存檔（path.unlink(missing_ok=True)）。
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.taipei_time import TAIPEI_TZ


# 子目錄名稱常數：上傳、重新壓縮、RAG
FOLDER_UPLOAD = "upload"
FOLDER_REPACK = "repack"
FOLDER_RAG = "rag"
# 上傳時未提供 person_id 時使用的目錄名
UPLOAD_DEFAULT_PERSON = "_"
# Bucket 內 metadata 物件路徑
_METADATA_KEY = "_metadata.json"


def _storage_safe_page_id(candidate: str, person_id: str | None = None) -> str:
    """
    Supabase Storage 物件 key 須為 ASCII；中文等非 ASCII 會觸發 InvalidKey。
    路徑片段含非 ASCII 時改以 generate_page_id 產生；純 ASCII 則沿用（含數字、底線、空白等）。
    """
    c = (candidate or "").strip()
    if not c or "/" in c or "\\" in c or len(c) > 255:
        return generate_page_id(person_id)
    if any(ord(ch) > 127 for ch in c):
        return generate_page_id(person_id)
    return c


def generate_page_id(person_id: str | None = None) -> str:
    """
    以 person_id 與目前電腦時間產生 page_id。
    格式：{person_id}_yymmddhhmmss。
    """
    safe = (person_id or "").strip() or "_"
    if "/" in safe or "\\" in safe:
        safe = "_"
    time_part = datetime.now(TAIPEI_TZ).strftime("%y%m%d%H%M%S")
    return f"{safe}_{time_part}"


def _get_bucket_name() -> str:
    """取得 Supabase Storage bucket 名稱（環境變數 SUPABASE_RAG_BUCKET，預設 "MyQuiz-ai"）。"""
    return os.environ.get("SUPABASE_RAG_BUCKET", "MyQuiz-ai")


def _get_storage():
    """取得 Supabase Storage bucket client。"""
    from utils.supabase import get_supabase
    return get_supabase().storage.from_(_get_bucket_name())


def _resolve_person_id(person_id: str | None) -> str:
    """將 person_id 轉為安全的目錄名。無效時回傳 UPLOAD_DEFAULT_PERSON。"""
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        return UPLOAD_DEFAULT_PERSON
    return pid


def _upload_object_basename(page_id: str) -> str:
    """
    Supabase Storage 物件 key 須為 ASCII 等合法字元；中文等 Unicode 檔名會觸發 InvalidKey。
    上傳 ZIP 在 bucket 內一律使用 {page_id}.zip。
    """
    return f"{page_id}.zip"


def _load_metadata() -> dict:
    """從 Supabase Storage 下載 _metadata.json；不存在或失敗時回傳空 dict。"""
    try:
        data = _get_storage().download(_METADATA_KEY)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def _save_metadata(data: dict) -> None:
    """將 metadata 上傳至 Supabase Storage（update 優先，失敗再 upload）。"""
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage = _get_storage()
    try:
        storage.update(_METADATA_KEY, content, {"content-type": "application/json"})
    except Exception:
        try:
            storage.upload(_METADATA_KEY, content, {"content-type": "application/json"})
        except Exception:
            pass


def _upload_to_bucket(storage_path: str, content: bytes) -> None:
    """上傳 bytes 至 Supabase Storage（upsert：update 優先，不存在則 upload）。"""
    storage = _get_storage()
    try:
        storage.update(storage_path, content, {"content-type": "application/zip"})
    except Exception:
        try:
            storage.remove([storage_path])
        except Exception:
            pass
        storage.upload(storage_path, content, {"content-type": "application/zip"})


def save_zip(
    contents: bytes,
    original_filename: str | None = None,
    folder: str = FOLDER_UPLOAD,
    person_id: str | None = None,
    page_id: str | None = None,
    parent_page_id: str | None = None,
) -> str:
    """
    將 ZIP 內容上傳至 Supabase Storage bucket，回傳 page_id。
    folder 可為 FOLDER_UPLOAD、FOLDER_REPACK、FOLDER_RAG。
    """
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
        # 顯示用原始檔名（可含中文）；bucket 內檔名固定為 {page_id}.zip
        display_name = (Path(original_filename).name if original_filename else "").strip() or _upload_object_basename(page_id)
        stored_name = _upload_object_basename(page_id)
        storage_path = f"{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}"
    else:
        stored_name = f"{page_id}.zip"
        storage_path = f"{pid}/{parent_page_id}/{folder}/{page_id}.zip"

    _upload_to_bucket(storage_path, contents)

    # 更新 metadata
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
    """
    從 Supabase Storage 下載檔案至系統暫存目錄，回傳 Path。
    失敗或檔案不存在時回傳 None。
    呼叫端負責用完後刪除暫存檔（path.unlink(missing_ok=True)）。
    """
    try:
        data = _get_storage().download(storage_path)
        if not data:
            return None
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="myquizai_dl_")
        os.close(fd)
        Path(tmp_path).write_bytes(data)
        return Path(tmp_path)
    except Exception:
        return None


def get_zip_path(page_id: str) -> Optional[Path]:
    """
    依 page_id 從 Supabase Storage 下載 ZIP 至暫存檔後回傳路徑；不存在則回傳 None。
    呼叫端負責用完後刪除暫存檔（path.unlink(missing_ok=True)）。
    """
    if not page_id or "/" in page_id or "\\" in page_id:
        return None

    meta = _load_metadata()
    entry = meta.get(page_id)

    if entry and isinstance(entry, dict):
        storage_path = entry.get("storage_path")
        if storage_path:
            return _download_to_temp(storage_path)

        # 從 metadata 欄位重建路徑
        folder = entry.get("folder", FOLDER_UPLOAD)
        pid = _resolve_person_id(entry.get("person_id"))
        parent_page_id = entry.get("parent_page_id")

        if folder == FOLDER_UPLOAD:
            stored_name = entry.get("stored_filename") or entry.get("filename") or f"{page_id}.zip"
            stored_name = Path(stored_name).name
            storage_path = f"{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}"
        elif parent_page_id:
            storage_path = f"{pid}/{parent_page_id}/{folder}/{page_id}.zip"
        else:
            return None

        return _download_to_temp(storage_path)

    return None


def get_zip_path_by_person(person_id: str, page_id: str) -> Optional[Path]:
    """
    依 person_id 與 page_id 從 Supabase Storage 下載上傳 ZIP 至暫存檔後回傳路徑。
    呼叫端負責用完後刪除暫存檔（path.unlink(missing_ok=True)）。
    """
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
        return _download_to_temp(f"{pid}/{page_id}/{FOLDER_UPLOAD}/{stored_name}")

    # 相容 fallback
    return _download_to_temp(f"{pid}/{page_id}/{FOLDER_UPLOAD}/{page_id}.zip")


def delete_tab_folder(person_id: str, page_id: str) -> bool:
    """
    刪除 Supabase Storage 中該 page_id 下的所有檔案（upload/repack/rag），
    以及以此 page_id 為 parent_page_id 的子項目。
    並清除 metadata 中對應的紀錄。
    回傳是否有刪除動作。
    """
    if not page_id or "/" in page_id or "\\" in page_id:
        return False
    pid = _resolve_person_id(person_id)

    storage = _get_storage()
    deleted_something = False

    # 刪除 upload / repack / rag 三個子目錄下的所有檔案
    for folder in [FOLDER_UPLOAD, FOLDER_REPACK, FOLDER_RAG]:
        try:
            files = storage.list(f"{pid}/{page_id}/{folder}")
            paths = [f"{pid}/{page_id}/{folder}/{f['name']}" for f in (files or []) if f.get("name")]
            if paths:
                storage.remove(paths)
                deleted_something = True
        except Exception:
            pass

    # 從 metadata 找出以此 page_id 為 parent_page_id 的子檔案並刪除
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

    # 清除 metadata 中該 page_id 及其子項目的紀錄
    to_remove = [k for k, v in meta.items() if k == page_id or (isinstance(v, dict) and v.get("parent_page_id") == page_id)]
    if to_remove:
        for k in to_remove:
            meta.pop(k, None)
        _save_metadata(meta)
        deleted_something = True

    return deleted_something
