"""
上傳檔案的儲存與解析。
ZIP 依類型存於 storage/{person_id}/{file_id}/ 之下：
- upload/：使用者上傳的 ZIP
- repack/：依資料夾重新壓縮的 ZIP
- rag/：RAG（FAISS）向量庫 ZIP
其他 API 可用 get_zip_path(file_id) 取得路徑讀取。
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path


def generate_file_id(person_id: str | None = None) -> str:
    """以 person_id 與目前電腦時間產生 file_id（格式：{person_id}_yymmddhhmmss）。"""
    safe = (person_id or "").strip() or "_"
    if "/" in safe or "\\" in safe:
        safe = "_"
    time_part = datetime.now().strftime("%y%m%d%H%M%S")
    return f"{safe}_{time_part}"

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
    parent_file_id: str | None = None,
) -> str:
    """
    將 ZIP 內容寫入後端儲存，回傳 file_id。
    folder 可為 FOLDER_UPLOAD（上傳）、FOLDER_REPACK（重新壓縮）、FOLDER_RAG（RAG 向量庫）。
    上傳時可傳入 file_id（由 API 指定）；repack/rag 可傳入 file_id（依 rag_list 命名，如 220222_220301），未傳入則產生 UUID。
    上傳存於 storage/{person_id}/{file_id}/upload/；無 person_id 時使用 storage/_/{file_id}/upload/。
    repack/rag 存於 storage/{person_id}/{parent_file_id}/repack/、storage/{person_id}/{parent_file_id}/rag/，需傳入 person_id 與 parent_file_id（上傳的 file_id）。
    其他 API 可用 get_zip_path(file_id) 取得檔案路徑後讀取。
    """
    if folder == FOLDER_UPLOAD:
        if file_id is not None and ("/" in file_id or "\\" in file_id or not file_id.strip()):
            raise ValueError("file_id 不可包含路徑字元且不可為空")
        if file_id:
            file_id = file_id.strip()
        else:
            file_id = generate_file_id(person_id)
    else:
        if not parent_file_id or "/" in parent_file_id or "\\" in parent_file_id:
            raise ValueError("repack/rag 需傳入 parent_file_id（上傳的 file_id）")
        parent_file_id = parent_file_id.strip()
        if file_id is not None and file_id.strip() and "/" not in file_id and "\\" not in file_id:
            file_id = file_id.strip()
        else:
            file_id = generate_file_id(person_id)

    if folder == FOLDER_UPLOAD:
        pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
        if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
            pid = UPLOAD_DEFAULT_PERSON
        target_dir = _storage_base() / pid / file_id / FOLDER_UPLOAD
        target_dir.mkdir(parents=True, exist_ok=True)
        # 上傳檔用原始檔名存檔，僅取 basename 避免路徑注入
        stored_name = (Path(original_filename).name if original_filename else "").strip() or f"{file_id}.zip"
        path = target_dir / stored_name
    else:
        pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
        if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
            pid = UPLOAD_DEFAULT_PERSON
        target_dir = _storage_base() / pid / parent_file_id / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{file_id}.zip"
    path.write_bytes(contents)
    meta = _load_metadata()
    meta[file_id] = {
        "filename": original_filename or (stored_name if folder == FOLDER_UPLOAD else f"{file_id}.zip"),
        "folder": folder,
    }
    if folder == FOLDER_UPLOAD:
        meta[file_id]["stored_filename"] = path.name
    if folder == FOLDER_UPLOAD:
        pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
        if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
            pid = UPLOAD_DEFAULT_PERSON
        meta[file_id]["person_id"] = pid
    else:
        meta[file_id]["person_id"] = pid
        meta[file_id]["parent_file_id"] = parent_file_id
    _save_metadata(meta)
    return file_id


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
    上傳：storage/{person_id}/{file_id}/upload/{file_id}.zip。
    repack/rag：storage/{person_id}/{parent_file_id}/repack|rag/{file_id}.zip。
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
    parent_file_id = None
    if entry is not None:
        if isinstance(entry, dict):
            folder = entry.get("folder", FOLDER_UPLOAD)
            person_id = entry.get("person_id")
            parent_file_id = entry.get("parent_file_id")
        else:
            folder = FOLDER_UPLOAD
    if folder is not None:
        if folder == FOLDER_UPLOAD:
            pid = person_id or file_id  # 舊版 metadata 無 person_id，用 file_id 當目錄
            # 上傳檔可能用原始檔名存檔，先查 metadata 的 stored_filename
            if isinstance(entry, dict):
                stored_name = entry.get("stored_filename") or entry.get("filename") or f"{file_id}.zip"
                stored_name = Path(stored_name).name if stored_name else f"{file_id}.zip"
            else:
                stored_name = f"{file_id}.zip"
            path = _storage_base() / pid / file_id / FOLDER_UPLOAD / stored_name
        else:
            # repack/rag：storage/{person_id}/{parent_file_id}/repack|rag/{file_id}.zip
            if person_id and parent_file_id:
                pid = person_id.strip() or UPLOAD_DEFAULT_PERSON
                if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
                    pid = UPLOAD_DEFAULT_PERSON
                path = _storage_base() / pid / parent_file_id / folder / f"{file_id}.zip"
            else:
                # 舊版：storage/repack/、storage/rag/
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


def get_zip_path_by_person(person_id: str, file_id: str) -> Path | None:
    """
    依 person_id 與 file_id 取得上傳 ZIP 路徑。上傳檔可能用原始檔名存於 upload/ 下。
    與 create-rag-zip 儲存位置一致。不存在則回傳 None。
    """
    if not file_id or "/" in file_id or "\\" in file_id:
        return None
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        pid = UPLOAD_DEFAULT_PERSON
    target_dir = _storage_base() / pid / file_id / FOLDER_UPLOAD
    # 先依 metadata 找實際檔名
    meta = _load_metadata()
    entry = meta.get(file_id)
    if isinstance(entry, dict):
        stored_name = entry.get("stored_filename") or entry.get("filename") or f"{file_id}.zip"
        stored_name = Path(stored_name).name if stored_name else f"{file_id}.zip"
    else:
        stored_name = f"{file_id}.zip"
    path = target_dir / stored_name
    if path.exists():
        return path
    # 相容舊資料：若無 metadata 或檔不存在，再試 file_id.zip
    fallback = target_dir / f"{file_id}.zip"
    return fallback if fallback.exists() else None


def get_file_folder_path(person_id: str, file_id: str) -> Path:
    """
    取得 storage/{person_id}/{file_id} 資料夾路徑（該 file_id 的 upload/repack/rag 皆在此下）。
    """
    if not file_id or "/" in file_id or "\\" in file_id:
        raise ValueError("file_id 不可包含路徑字元且不可為空")
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        pid = UPLOAD_DEFAULT_PERSON
    return _storage_base() / pid / file_id.strip()


def delete_file_folder(person_id: str, file_id: str) -> bool:
    """
    刪除 storage/{person_id}/{file_id}/ 整個資料夾（含 upload、repack、rag）。
    僅在路徑位於儲存根目錄下時執行，避免路徑穿越。並清除 metadata 中該 file_id 與其下 repack/rag 的紀錄。
    回傳是否已刪除（資料夾存在且已刪除為 True）。
    """
    target = get_file_folder_path(person_id, file_id)
    base = _storage_base().resolve()
    target_resolved = target.resolve()
    if not target_resolved.is_relative_to(base) or target_resolved == base:
        return False
    if not target.exists() or not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    # 清除 metadata：該 file_id（upload）與 parent_file_id 為此 file_id 的 repack/rag
    meta = _load_metadata()
    to_remove = [fid for fid, entry in meta.items() if fid == file_id or (isinstance(entry, dict) and entry.get("parent_file_id") == file_id)]
    for fid in to_remove:
        meta.pop(fid, None)
    _save_metadata(meta)
    return True
