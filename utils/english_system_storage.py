"""
English System 音檔上傳至 Supabase Storage（與 RAG 相同階層概念）。

Bucket 名稱：環境變數 SUPABASE_ENGLISH_BUCKET（預設 "english_system"）。

路徑結構（對齊 zip_storage 之 upload）：
  {person_id}/{system_tab_id}/upload/{system_tab_id}{副檔名}
  物件 key 僅用 system_tab_id + 副檔名（ASCII）；原始檔名寫入 _metadata.json。

_metadata.json 位於 bucket 根目錄，以 system_tab_id 為鍵（類似 rag 以 tab_id 為鍵）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from utils.zip_storage import FOLDER_UPLOAD, UPLOAD_DEFAULT_PERSON

_METADATA_KEY = "_metadata.json"


def _get_bucket_name() -> str:
    return os.environ.get("SUPABASE_ENGLISH_BUCKET", "english_system")


def _get_storage():
    from utils.supabase_client import get_supabase

    return get_supabase().storage.from_(_get_bucket_name())


def _resolve_person_id(person_id: str | None) -> str:
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        return UPLOAD_DEFAULT_PERSON
    return pid


def _stored_basename(system_tab_id: str, suffix: str) -> str:
    """bucket 內檔名：{system_tab_id}.mp3 等（suffix 含點）。"""
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{system_tab_id}{ext}"


def _content_type_for_suffix(suffix: str) -> str:
    s = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".mp4": "audio/mp4",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
    }.get(s, "application/octet-stream")


def _load_metadata() -> dict[str, Any]:
    try:
        data = _get_storage().download(_METADATA_KEY)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def _save_metadata(data: dict[str, Any]) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage = _get_storage()
    try:
        storage.update(_METADATA_KEY, content, {"content-type": "application/json"})
    except Exception:
        try:
            storage.upload(_METADATA_KEY, content, {"content-type": "application/json"})
        except Exception:
            pass


def _upload_bytes(storage_path: str, content: bytes, content_type: str) -> None:
    storage = _get_storage()
    try:
        storage.update(storage_path, content, {"content-type": content_type})
    except Exception:
        try:
            storage.remove([storage_path])
        except Exception:
            pass
        storage.upload(storage_path, content, {"content-type": content_type})


def save_english_system_upload_audio(
    contents: bytes,
    *,
    person_id: str,
    system_tab_id: str,
    original_filename: str | None,
    suffix: str,
) -> dict[str, str]:
    """
    上傳音訊至 english_system bucket（upload 區）。
    回傳 bucket、storage_path、stored_filename、display_filename。
    """
    fid = (system_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise ValueError("system_tab_id 不可為空且不可含路徑字元")
    pid = _resolve_person_id(person_id)
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    stored_name = _stored_basename(fid, ext)
    storage_path = f"{pid}/{fid}/{FOLDER_UPLOAD}/{stored_name}"
    display_name = (Path(original_filename).name if original_filename else "").strip() or stored_name

    _upload_bytes(storage_path, contents, _content_type_for_suffix(ext))

    meta = _load_metadata()
    meta[fid] = {
        "filename": display_name,
        "folder": FOLDER_UPLOAD,
        "person_id": pid,
        "storage_path": storage_path,
        "stored_filename": stored_name,
        "kind": "english_audio",
    }
    _save_metadata(meta)

    return {
        "bucket": _get_bucket_name(),
        "storage_path": storage_path,
        "stored_filename": stored_name,
        "display_filename": display_name,
    }
