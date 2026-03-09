"""
上傳檔案的儲存與解析模組。
ZIP 依類型存於 storage/{person_id}/{tab_id}/ 之下：
- upload/：使用者上傳的 ZIP
- repack/：依資料夾重新壓縮的 ZIP
- rag/：RAG（FAISS）向量庫 ZIP
其他 API 可用 get_zip_path(tab_id) 取得路徑讀取。
"""

# 引入 json 模組，用於讀寫 metadata
import json
# 引入 os 模組，用於環境變數與路徑
import os
# 引入 shutil 模組，用於刪除目錄
import shutil
# 引入 datetime 用於產生時間戳記
from datetime import datetime
# 引入 Path 用於路徑操作
from pathlib import Path


def generate_tab_id(person_id: str | None = None) -> str:
    """
    以 person_id 與目前電腦時間產生 tab_id。
    格式：{person_id}_yymmddhhmmss。
    """
    # 若 person_id 為空，使用 "_"；若含路徑字元則改為 "_"
    safe = (person_id or "").strip() or "_"
    # 若含 / 或 \，改為 "_" 避免路徑注入
    if "/" in safe or "\\" in safe:
        safe = "_"
    # 取得目前時間的 yymmddhhmmss 格式
    time_part = datetime.now().strftime("%y%m%d%H%M%S")
    return f"{safe}_{time_part}"

# 子目錄名稱常數：上傳、重新壓縮、RAG
FOLDER_UPLOAD = "upload"  # 使用者上傳的 ZIP
FOLDER_REPACK = "repack"  # 重新壓縮的 ZIP
FOLDER_RAG = "rag"  # RAG 向量庫 ZIP
# 上傳時未提供 person_id 時使用的目錄名
UPLOAD_DEFAULT_PERSON = "_"


def _storage_base() -> Path:
    """
    取得儲存根目錄。
    可由環境變數 ZIP_STORAGE_DIR 指定，預設為專案下的 storage/。
    """
    # 從環境變數取得儲存根目錄，預設 "storage"
    base = os.environ.get("ZIP_STORAGE_DIR", "storage")
    # 轉成 Path 物件
    path = Path(base)
    # 若目錄不存在則建立
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_person_id(person_id: str | None) -> str:
    """
    將 person_id 轉為安全的目錄名。
    無效時（含路徑字元、空字串、.、..）回傳 UPLOAD_DEFAULT_PERSON。
    """
    # 去除空白，空則用 UPLOAD_DEFAULT_PERSON
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    # 若含路徑字元或為 .、..，回傳預設值
    if "/" in pid or "\\" in pid or pid in ("", ".", ".."):
        return UPLOAD_DEFAULT_PERSON
    return pid


def _folder_dir(folder: str) -> Path:
    """
    取得指定類型子目錄（upload / repack / rag）的路徑。
    不存在時會建立。
    """
    # 組裝路徑：storage_base/folder
    path = _storage_base() / folder
    # 若不存在則建立
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path() -> Path:
    """取得 metadata 檔案路徑（_storage_base/_metadata.json）。"""
    return _storage_base() / "_metadata.json"


def _load_metadata() -> dict:
    """載入 metadata JSON；不存在或解析失敗時回傳空 dict。"""
    # 取得 metadata 檔案路徑
    p = _metadata_path()
    # 若不存在，回傳空 dict
    if not p.exists():
        return {}
    try:
        # 讀取並解析 JSON
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_metadata(data: dict) -> None:
    """將 metadata 寫入 JSON 檔案。"""
    _metadata_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_zip(
    contents: bytes,
    original_filename: str | None = None,
    folder: str = FOLDER_UPLOAD,
    person_id: str | None = None,
    tab_id: str | None = None,
    parent_tab_id: str | None = None,
) -> str:
    """
    將 ZIP 內容寫入後端儲存，回傳 tab_id。
    folder 可為 FOLDER_UPLOAD（上傳）、FOLDER_REPACK（重新壓縮）、FOLDER_RAG（RAG 向量庫）。
    上傳時可傳入 tab_id（由 API 指定）；repack/rag 可傳入 tab_id（依 rag_list 命名），未傳入則產生 UUID。
    """
    if folder == FOLDER_UPLOAD:
        # 上傳模式：檢查 tab_id 是否有效
        if tab_id is not None and ("/" in tab_id or "\\" in tab_id or not tab_id.strip()):
            raise ValueError("tab_id 不可包含路徑字元且不可為空")
        if tab_id:
            tab_id = tab_id.strip()  # 去除空白
        else:
            tab_id = generate_tab_id(person_id)  # 產生新 tab_id
    else:
        # repack/rag 模式：必須有 parent_tab_id
        if not parent_tab_id or "/" in parent_tab_id or "\\" in parent_tab_id:
            raise ValueError("repack/rag 需傳入 parent_tab_id（上傳的 tab_id）")
        parent_tab_id = parent_tab_id.strip()
        if tab_id is not None and tab_id.strip() and "/" not in tab_id and "\\" not in tab_id:
            tab_id = tab_id.strip()  # 使用傳入的 tab_id
        else:
            tab_id = generate_tab_id(person_id)  # 產生新 tab_id

    if folder == FOLDER_UPLOAD:
        # 上傳：路徑為 storage/{person_id}/{tab_id}/upload/
        pid = _resolve_person_id(person_id)
        target_dir = _storage_base() / pid / tab_id / FOLDER_UPLOAD
        target_dir.mkdir(parents=True, exist_ok=True)
        # 上傳檔用原始檔名存檔，僅取 basename 避免路徑注入
        stored_name = (Path(original_filename).name if original_filename else "").strip() or f"{tab_id}.zip"
        path = target_dir / stored_name
    else:
        # repack/rag：路徑為 storage/{person_id}/{parent_tab_id}/{folder}/{tab_id}.zip
        pid = _resolve_person_id(person_id)
        target_dir = _storage_base() / pid / parent_tab_id / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{tab_id}.zip"
    # 寫入 ZIP 內容
    path.write_bytes(contents)
    # 更新 metadata
    meta = _load_metadata()
    meta[tab_id] = {
        "filename": original_filename or (stored_name if folder == FOLDER_UPLOAD else f"{tab_id}.zip"),
        "folder": folder,
    }
    if folder == FOLDER_UPLOAD:
        meta[tab_id]["stored_filename"] = path.name  # 實際存檔檔名
    meta[tab_id]["person_id"] = pid
    if folder != FOLDER_UPLOAD:
        meta[tab_id]["parent_tab_id"] = parent_tab_id  # repack/rag 需記錄 parent
    _save_metadata(meta)
    return tab_id


def get_zip_path(tab_id: str) -> Path | None:
    """
    依 tab_id 取得已儲存的 ZIP 檔案路徑；不存在則回傳 None。
    上傳：storage/{person_id}/{tab_id}/upload/{stored_filename}。
    repack/rag：storage/{person_id}/{parent_tab_id}/repack|rag/{tab_id}.zip。
    """
    # 若 tab_id 無效，回傳 None
    if not tab_id or "/" in tab_id or "\\" in tab_id:
        return None
    meta = _load_metadata()
    entry = meta.get(tab_id)
    folder = None
    person_id = None
    parent_tab_id = None
    if entry is not None:
        if isinstance(entry, dict):
            folder = entry.get("folder", FOLDER_UPLOAD)
            person_id = entry.get("person_id")
            parent_tab_id = entry.get("parent_tab_id")
        else:
            folder = FOLDER_UPLOAD  # 舊版格式
    if folder is not None:
        if folder == FOLDER_UPLOAD:
            pid = person_id or tab_id  # 舊版 metadata 無 person_id，用 tab_id 當目錄
            if isinstance(entry, dict):
                stored_name = entry.get("stored_filename") or entry.get("filename") or f"{tab_id}.zip"
                stored_name = Path(stored_name).name if stored_name else f"{tab_id}.zip"
            else:
                stored_name = f"{tab_id}.zip"
            path = _storage_base() / pid / tab_id / FOLDER_UPLOAD / stored_name
        else:
            if person_id and parent_tab_id:
                pid = _resolve_person_id(person_id)
                path = _storage_base() / pid / parent_tab_id / folder / f"{tab_id}.zip"
            else:
                path = _folder_dir(folder) / f"{tab_id}.zip"  # 舊版路徑
        if path.exists():
            return path
    # 舊版相容：storage/{tab_id}/upload/
    upload_by_tab_id = _storage_base() / tab_id / FOLDER_UPLOAD / f"{tab_id}.zip"
    if upload_by_tab_id.exists():
        return upload_by_tab_id
    path = _folder_dir(FOLDER_UPLOAD) / f"{tab_id}.zip"
    if path.exists():
        return path
    legacy = _storage_base() / f"{tab_id}.zip"
    return legacy if legacy.exists() else None


def get_zip_path_by_person(person_id: str, tab_id: str) -> Path | None:
    """
    依 person_id 與 tab_id 取得上傳 ZIP 路徑。
    上傳檔可能用原始檔名存於 upload/ 下；不存在則回傳 None。
    """
    if not tab_id or "/" in tab_id or "\\" in tab_id:
        return None
    pid = _resolve_person_id(person_id)
    target_dir = _storage_base() / pid / tab_id / FOLDER_UPLOAD
    meta = _load_metadata()
    entry = meta.get(tab_id)
    if isinstance(entry, dict):
        stored_name = entry.get("stored_filename") or entry.get("filename") or f"{tab_id}.zip"
        stored_name = Path(stored_name).name if stored_name else f"{tab_id}.zip"
    else:
        stored_name = f"{tab_id}.zip"
    path = target_dir / stored_name
    if path.exists():
        return path
    fallback = target_dir / f"{tab_id}.zip"  # 相容：試 tab_id.zip
    return fallback if fallback.exists() else None


def get_tab_folder_path(person_id: str, tab_id: str) -> Path:
    """
    取得 storage/{person_id}/{tab_id} 資料夾路徑。
    該 tab_id 的 upload/repack/rag 皆在此目錄下。
    """
    if not tab_id or "/" in tab_id or "\\" in tab_id:
        raise ValueError("tab_id 不可包含路徑字元且不可為空")
    return _storage_base() / _resolve_person_id(person_id) / tab_id.strip()


def delete_tab_folder(person_id: str, tab_id: str) -> bool:
    """
    刪除 storage/{person_id}/{tab_id}/ 整個資料夾（含 upload、repack、rag）。
    僅在路徑位於儲存根目錄下時執行，避免路徑穿越。
    並清除 metadata 中該 tab_id 與其下 repack/rag 的紀錄。
    回傳是否已刪除。
    """
    target = get_tab_folder_path(person_id, tab_id)
    base = _storage_base().resolve()
    target_resolved = target.resolve()
    # 檢查 target 是否在 base 之下且不等於 base（防止路徑穿越）
    if not target_resolved.is_relative_to(base) or target_resolved == base:
        return False
    if not target.exists() or not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    # 清除 metadata 中該 tab_id 及 parent_tab_id 為此 tab_id 的 repack/rag
    meta = _load_metadata()
    to_remove = [fid for fid, entry in meta.items() if fid == tab_id or (isinstance(entry, dict) and entry.get("parent_tab_id") == tab_id)]
    for fid in to_remove:
        meta.pop(fid, None)
    _save_metadata(meta)  # 寫回 metadata
    return True
