"""
ZIP 與 RAG 相關 API 模組。
提供：
- GET /rag/tabs：列出 Rag 表（含 units→quizzes）；僅回傳 query person_id 與 Rag.person_id 相同之列；query `local` 篩選 Rag.local，未傳時依連線是否本機判定；回傳依 created_at 舊→新
- GET /rag/units：依 rag_tab_id 列出 Rag_Unit（含 quizzes）
- POST /rag/tab/create：建立一筆 Rag（可傳 local）
- PUT /rag/tab/tab-name：更新既有 Rag 的 tab_name（body：rag_id、tab_name）
- POST /rag/tab/upload-zip：上傳 ZIP
- POST /rag/tab/build-rag-zip：依 unit_list 打包；unit_type=1 且允許 FAISS 時建向量庫上傳 rag；unit_type=2/3/4 時 repack 照舊，rag 區改上傳「逐字稿全文之單檔 transcript.md」所包成的 ZIP（非 repack 複製；**unit_type=2** 時 **text_file_name** 記錄上傳 ZIP 內來源文字檔檔名）；可選 body.build_faiss 覆寫；回應 NDJSON。POST /rag/tab/build-rag-zip-stream 為別名
- POST /rag/tab/delete/{rag_tab_id}：依 rag_tab_id 軟刪除 Rag 及其 Rag_Unit，並刪除儲存（須傳 query person_id）
- PUT /rag/tab/unit/unit-name：更新 Rag_Unit 的 unit_name（body：rag_unit_id、unit_name）
- POST /rag/tab/unit/quiz/create：body `rag_tab_id`、`rag_unit_id` 定位 Rag_Unit 後新增一筆 Rag_Quiz（無 LLM）；`rag_quiz_id` 由資料庫產生。
"""

import io
import json
import os
import time
import logging
import uuid
import tempfile
import zipfile

from storage3.exceptions import StorageApiError
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Path as PathParam, Query, Request
from fastapi.responses import StreamingResponse

from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from utils.datetime_utils import now_taipei_iso, to_taipei_iso
from utils.json_utils import to_json_safe
from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
)
from utils.zip_storage import (
    save_zip,
    get_zip_path,
    get_zip_path_by_person,
    delete_tab_folder,
    FOLDER_UPLOAD,
    FOLDER_REPACK,
    FOLDER_RAG,
)
from utils.supabase_client import get_supabase
from utils.db_tables import USER_TABLE
from utils.rag_exam_setting import is_localhost_request
from utils.rag_transcript_from_upload_zip import (
    build_transcript_md_zip_bytes,
    extract_transcript_for_rag_build,
)
from routers.grade import _quiz_grade_from_answer_critique

router = APIRouter(prefix="/rag", tags=["rag"])

RAG_SELECT_ALL = "*"

BYTES_PER_MB = 1024 * 1024

# Rag_Unit.unit_type（PostgreSQL smallint）
# 0=未選／預設；1=rag（ZIP 建向量）；2=文字；3=mp3；4=youtube
RAG_UNIT_TYPE_DEFAULT = 0
RAG_UNIT_TYPE_RAG = 1
RAG_UNIT_TYPE_TEXT = 2
RAG_UNIT_TYPE_MP3 = 3
RAG_UNIT_TYPE_YOUTUBE = 4
# 舊註解「ZIP 打包建置」同義於 rag
RAG_UNIT_TYPE_ZIP_BUILD = RAG_UNIT_TYPE_RAG


def _bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / BYTES_PER_MB


def _fetch_user_llm_key_and_user_type(person_id: str) -> tuple[str | None, int]:
    """依 person_id 自 User 表取得 llm_api_key、user_type；無列時回傳 (None, 0)。"""
    pid = (person_id or "").strip()
    if not pid:
        return None, 0
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("llm_api_key, user_type")
            .eq("person_id", pid)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None, 0
        row = resp.data[0]
        key = (row.get("llm_api_key") or "").strip() or None
        ut = row.get("user_type")
        try:
            ut_int = int(ut) if ut is not None else 0
        except (TypeError, ValueError):
            ut_int = 0
        return key, ut_int
    except Exception:
        return None, 0


def _unit_types_per_task(unit_types_csv: str, task_count: int) -> list[int]:
    """與 unit_list 逗號分段對齊；值限制在 0–4，缺漏為 0。"""
    parts = [p.strip() for p in unit_types_csv.split(",")] if (unit_types_csv or "").strip() else []
    out: list[int] = []
    for i in range(task_count):
        v = 0
        if i < len(parts) and parts[i]:
            try:
                v = int(parts[i])
            except ValueError:
                v = 0
        if v < 0 or v > 4:
            v = 0
        out.append(v)
    return out


def _rag_default_row(
    rag_tab_id: str,
    *,
    tab_name: str | None = None,
    person_id: str | None = None,
    file_metadata: Any = None,
    local: bool = False,
) -> dict[str, Any]:
    """Rag 表一筆新增時的預設欄位；鍵順序同 public.Rag（rag_tab_id→…；不含 rag_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    row: dict[str, Any] = {
        "rag_tab_id": rag_tab_id,
        "tab_name": tab_name if tab_name is not None else "",
        "person_id": person_id if person_id is not None else "",
        "file_metadata": file_metadata,
    }
    # rag_metadata 在部分環境可能尚未完成 migration；建立時先不主動寫入，避免 500。
    row["chunk_size"] = 1000
    row["chunk_overlap"] = 200
    row["local"] = local
    row["deleted"] = False
    row["created_at"] = ts
    row["updated_at"] = ts
    return row


def _rag_unit_default_row(
    rag_tab_id: str,
    person_id: str,
    *,
    unit_name: str = "",
    unit_type: int = RAG_UNIT_TYPE_RAG,
    repack_file_name: str = "",
    rag_file_name: str = "",
    rag_file_size: float = 0.0,
    transcription: str = "",
    text_file_name: str = "",
    mp3_file_name: str = "",
    youtube_url: str = "",
) -> dict[str, Any]:
    """Rag_Unit 表一筆新增時的預設欄位。"""
    ts = now_taipei_iso()
    return {
        "rag_tab_id": rag_tab_id,
        "person_id": person_id,
        "unit_name": unit_name,
        "unit_type": unit_type,
        "repack_file_name": repack_file_name,
        "rag_file_name": rag_file_name,
        "rag_file_size": rag_file_size,
        "transcription": transcription,
        "text_file_name": text_file_name,
        "mp3_file_name": mp3_file_name,
        "youtube_url": youtube_url,
        "deleted": False,
        "created_at": ts,
        "updated_at": ts,
    }


def _rag_table_select(
    select_columns: str = "*",
    exclude_deleted: bool = False,
    *,
    local_match: bool | None = None,
) -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。exclude_deleted=True 時僅回傳 deleted=False。local_match 若指定則僅回傳 Rag.local 與其相符的列。依 created_at 升序（舊→新）。"""
    supabase = get_supabase()
    q = supabase.table("Rag").select(select_columns)
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    q = q.order("created_at", desc=False)
    resp = q.execute()
    return resp.data or []


def _units_by_rag_tab_ids(rag_tab_ids: list[str]) -> dict[str, list[dict]]:
    """依 rag_tab_id 查詢 Rag_Unit 表，回傳 rag_tab_id -> list of unit 列。僅回傳 deleted=False。依 created_at 升序。"""
    if not rag_tab_ids:
        return {}
    supabase = get_supabase()
    resp = (
        supabase.table("Rag_Unit")
        .select("*")
        .in_("rag_tab_id", rag_tab_ids)
        .eq("deleted", False)
        .order("created_at", desc=False)
        .execute()
    )
    rows = resp.data or []
    out: dict[str, list[dict]] = {tid: [] for tid in rag_tab_ids}
    for row in rows:
        tid = row.get("rag_tab_id")
        if tid is not None:
            out.setdefault(tid, []).append(row)
    return out


def _quizzes_by_rag_unit_ids(rag_unit_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_unit_id 查詢 Rag_Quiz 表，回傳 rag_unit_id -> list of quiz 列。僅回傳 deleted=False。"""
    if not rag_unit_ids:
        return {}
    supabase = get_supabase()
    resp = (
        supabase.table("Rag_Quiz")
        .select("*")
        .in_("rag_unit_id", rag_unit_ids)
        .eq("deleted", False)
        .execute()
    )
    rows = resp.data or []
    out: dict[int, list[dict]] = {uid: [] for uid in rag_unit_ids}
    for row in rows:
        uid = row.get("rag_unit_id")
        if uid is not None:
            try:
                out.setdefault(int(uid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


class ListRagResponse(BaseModel):
    """GET /rag/tabs 回應：每筆 Rag 含 units（Rag_Unit），每個 unit 含 quizzes（Rag_Quiz）。"""
    rags: list[dict]
    count: int


class CreateRagRequest(BaseModel):
    """POST /rag/tab/create：欄位順序同 public.Rag（rag_tab_id, tab_name, person_id, local）。"""
    rag_tab_id: str = Field(..., description="Rag 的 tab 識別，對應 Rag 表 rag_tab_id 欄位")
    tab_name: str = Field(..., description="Rag 顯示名稱，寫入 Rag 表 tab_name 欄位")
    person_id: str = Field(..., description="使用者/路徑識別")
    local: bool = Field(False, description="是否為本機 RAG，寫入 Rag 表 local 欄位")


class UpdateRagUnitNameRequest(BaseModel):
    """PUT /rag/tab/tab-name：請求僅含 rag_id（主鍵）、tab_name；勿傳 rag_tab_id。"""
    rag_id: int = Field(..., description="Rag 表主鍵（整數）；辨識請用 rag_id，非 rag_tab_id")
    tab_name: str = Field(..., description="新的顯示名稱，寫入 Rag 表 tab_name 欄位")


class PackRequest(BaseModel):
    """欄位順序對應 public.Rag 中本請求會更新的區段：rag_tab_id, person_id, unit_list（用於指定要打包的資料夾，結果存入 Rag_Unit）, chunk_size, chunk_overlap。"""
    rag_tab_id: str
    person_id: str
    unit_list: str  # 指定要打包的資料夾；例："220222+220301"（加號=同一 ZIP 多資料夾）；結果存入 Rag_Unit 表
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 與 unit_list 同樣以逗號分段對齊每一個打包任務；0=未選、1=rag、2=文字、3=mp3、4=youtube；省略或不足處視為 0
    unit_types: str = ""
    # 省略=None：依 User.user_type==1 決定是否建 FAISS；False=強制不建向量（rag 僅上傳與 repack 相同之 ZIP）；True=強制建 FAISS（須 llm_api_key）
    build_faiss: bool | None = Field(
        default=None,
        description="省略時依 User.user_type；False 時僅複製 repack 至 rag；True 時強制建向量 RAG ZIP",
    )


class UpdateRagUnitUnitNameRequest(BaseModel):
    """PUT /rag/tab/unit/unit-name：更新 Rag_Unit 的 unit_name。"""
    rag_unit_id: int = Field(..., description="Rag_Unit 表主鍵")
    unit_name: str = Field(..., description="新的 unit_name")


class InsertRagQuizRowRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/create：欄位順序對齊 public.Rag_Quiz 之關聯欄（rag_tab_id、rag_unit_id）。
    `rag_tab_id` 與 `rag_unit_id` 二擇一定位 Rag_Unit：
    - `rag_unit_id > 0`：以主鍵載入；若同傳 `rag_tab_id`（非空）則須與該列一致。
    - `rag_unit_id == 0`：`rag_tab_id`（非空）須在該名下**唯一**一筆未刪除之 Rag_Unit，否則 400。
    """

    rag_tab_id: str = Field("", description="Rag tab 識別；與 rag_unit_id 併用見上")
    rag_unit_id: int = Field(0, ge=0, description="Rag_Unit 主鍵；0 表示改由 rag_tab_id 唯一解析")


_MIN_RAG_ZIP_BYTES = 22
_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS = 80
_SOURCE_UPLOAD_ZIP_SLEEP_SEC = 0.45
_RAG_ZIP_VERIFY_MAX_ATTEMPTS = 18
_RAG_ZIP_VERIFY_SLEEP_INITIAL = 0.35
_RAG_ZIP_VERIFY_SLEEP_MAX = 3.0
_RAG_UNIT_FULL_BUILD_ATTEMPTS = 3
_RAG_UNIT_FULL_BUILD_RETRY_SLEEP_BASE = 0.65


def _try_read_upload_zip_once(pid: str, rag_tab_id: str) -> Path | None:
    path = get_zip_path(rag_tab_id) or get_zip_path_by_person(pid, rag_tab_id)
    if not path or not path.exists():
        return None
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.namelist()
        return path
    except zipfile.BadZipFile:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _fetch_source_upload_zip_with_retries(pid: str, rag_tab_id: str) -> Path:
    last_fail = "無法取得路徑或下載失敗"
    for attempt in range(_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS):
        path = _try_read_upload_zip_once(pid, rag_tab_id)
        if path is not None:
            return path
        last_fail = f"第 {attempt + 1}/{_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS} 次仍無法讀取上傳 ZIP"
        time.sleep(_SOURCE_UPLOAD_ZIP_SLEEP_SEC)
    raise HTTPException(
        status_code=503,
        detail=(
            f"多次重試後仍無法讀取上傳的 ZIP（rag_tab_id={rag_tab_id}）。"
            f"若剛上傳完請稍後再試；{last_fail}"
        ),
    )


def _verify_saved_rag_zip_readable(rag_zip_tab_id: str) -> str | None:
    delay = _RAG_ZIP_VERIFY_SLEEP_INITIAL
    last_err = "RAG ZIP 上傳後無法從儲存讀回驗證"
    for _ in range(_RAG_ZIP_VERIFY_MAX_ATTEMPTS):
        verify_path = get_zip_path(rag_zip_tab_id)
        if not verify_path or not verify_path.exists():
            last_err = "RAG ZIP 上傳後無法從儲存讀回驗證"
            time.sleep(delay)
            delay = min(delay * 1.3, _RAG_ZIP_VERIFY_SLEEP_MAX)
            continue
        try:
            sz = verify_path.stat().st_size
            if sz < _MIN_RAG_ZIP_BYTES:
                last_err = "RAG ZIP 驗證失敗（讀回檔案過小或損毀）"
            else:
                return None
        finally:
            try:
                verify_path.unlink(missing_ok=True)
            except Exception:
                pass
        time.sleep(delay)
        delay = min(delay * 1.3, _RAG_ZIP_VERIFY_SLEEP_MAX)
    return last_err


def _build_one_rag_zip_output_item(
    body: PackRequest,
    pid: str,
    api_key: str,
    zip_bytes: bytes,
    filename: str,
    *,
    do_rag: bool = True,
    unit_type: int = RAG_UNIT_TYPE_DEFAULT,
) -> dict[str, Any]:
    """
    將單一 repack 單元上傳至 repack。
    do_rag 為 True：另以 FAISS 建 RAG ZIP 上傳至 rag。
    do_rag 為 False 且 unit_type 為 2/3/4：repack 照舊，rag 區上傳逐字稿之 transcript.md ZIP（rag_mode=transcript_md），output 含 transcript_plain 與對應檔名／url 欄位。
    do_rag 為 False 且非上述：將與 repack 相同內容複製至 rag（rag_mode=repack_copy）。
    註：bucket 內檔名仍為 stem_rag.zip；請看 output.rag_mode。
    回傳與 build-rag-zip outputs[] 單筆相同結構；失敗時含 rag_error。
    """
    repack_tab_id = Path(filename).stem if filename else None
    if not repack_tab_id or "/" in repack_tab_id or "\\" in repack_tab_id:
        repack_tab_id = str(uuid.uuid4())

    if not do_rag:
        item_zip: dict[str, Any] = {
            "filename": filename,
            "unit_name": repack_tab_id,
            "repack_filename": f"{repack_tab_id}.zip",
            "rag_filename": "",
            "unit_type": unit_type,
            "rag_mode": "repack_copy",
            "transcript_plain": "",
            "text_file_name": "",
            "mp3_file_name": "",
            "youtube_url": "",
        }
        try:
            tab_id = save_zip(
                zip_bytes,
                filename,
                folder=FOLDER_REPACK,
                person_id=pid,
                parent_tab_id=body.rag_tab_id,
                tab_id=repack_tab_id,
            )
            # unit_name 維持為資料夾顯示名（可含中文）；bucket 內檔名為 ASCII tab_id
            item_zip["repack_filename"] = f"{tab_id}.zip"
            rag_zip_tab_id = f"{tab_id}_rag"
            item_zip["rag_filename"] = f"{rag_zip_tab_id}.zip"
            rag_payload: bytes | None = None
            if unit_type in (RAG_UNIT_TYPE_TEXT, RAG_UNIT_TYPE_MP3, RAG_UNIT_TYPE_YOUTUBE):
                item_zip["rag_mode"] = "transcript_md"
                try:
                    extracted = extract_transcript_for_rag_build(zip_bytes, unit_type)
                except ValueError as e:
                    item_zip["rag_error"] = str(e)
                else:
                    item_zip["transcript_plain"] = extracted.get("transcript") or ""
                    item_zip["text_file_name"] = extracted.get("text_file_name") or ""
                    item_zip["mp3_file_name"] = extracted.get("mp3_file_name") or ""
                    item_zip["youtube_url"] = extracted.get("youtube_url") or ""
                    rag_payload = build_transcript_md_zip_bytes(item_zip["transcript_plain"])
                    save_zip(
                        rag_payload,
                        f"{tab_id}.zip",
                        folder=FOLDER_RAG,
                        person_id=pid,
                        parent_tab_id=body.rag_tab_id,
                        tab_id=rag_zip_tab_id,
                    )
            else:
                rag_payload = zip_bytes
                save_zip(
                    zip_bytes,
                    f"{tab_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_tab_id=body.rag_tab_id,
                    tab_id=rag_zip_tab_id,
                )
            if not item_zip.get("rag_error"):
                verify_err = _verify_saved_rag_zip_readable(rag_zip_tab_id)
                if verify_err:
                    item_zip["rag_error"] = verify_err
            item_zip["file_size"] = _bytes_to_mb(
                len(rag_payload) if rag_payload is not None else len(zip_bytes)
            )
        except Exception as e:
            item_zip["rag_error"] = str(e)
        return item_zip

    last_item: dict[str, Any] = {}
    for attempt in range(_RAG_UNIT_FULL_BUILD_ATTEMPTS):
        item: dict[str, Any] = {
            "filename": filename,
            "unit_name": repack_tab_id,
            "repack_filename": f"{repack_tab_id}.zip",
            "rag_filename": f"{repack_tab_id}_rag.zip",
            "unit_type": unit_type,
            "rag_mode": "faiss",
        }
        rag_bytes_out: bytes | None = None
        try:
            from utils.rag_faiss_zip import make_rag_zip_from_zip_path

            tab_id = save_zip(
                zip_bytes,
                filename,
                folder=FOLDER_REPACK,
                person_id=pid,
                parent_tab_id=body.rag_tab_id,
                tab_id=repack_tab_id,
            )
            rag_zip_tab_id = f"{tab_id}_rag"
            # unit_name 維持為資料夾顯示名（可含中文）；向量檔路徑依 ASCII tab_id
            item["repack_filename"] = f"{tab_id}.zip"
            item["rag_filename"] = f"{rag_zip_tab_id}.zip"

            fd, repack_tmp = tempfile.mkstemp(suffix=".zip", prefix="myquizai_repack_")
            os.close(fd)
            repack_local = Path(repack_tmp)
            try:
                repack_local.write_bytes(zip_bytes)
                rag_bytes_out = make_rag_zip_from_zip_path(
                    repack_local,
                    api_key,
                    chunk_size=body.chunk_size,
                    chunk_overlap=body.chunk_overlap,
                    unit_type=unit_type,
                )
            finally:
                try:
                    repack_local.unlink(missing_ok=True)
                except Exception:
                    pass

            if not rag_bytes_out or len(rag_bytes_out) < _MIN_RAG_ZIP_BYTES:
                item["rag_error"] = "RAG ZIP 產物無效或過小（未產生可用向量庫 ZIP）"
            else:
                save_zip(
                    rag_bytes_out,
                    f"{tab_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_tab_id=body.rag_tab_id,
                    tab_id=rag_zip_tab_id,
                )
                verify_err = _verify_saved_rag_zip_readable(rag_zip_tab_id)
                if verify_err:
                    item["rag_error"] = verify_err
        except ValueError as e:
            item["rag_error"] = str(e)
        except Exception as e:
            item["rag_error"] = str(e)

        item["file_size"] = _bytes_to_mb(len(rag_bytes_out) if rag_bytes_out is not None else len(zip_bytes))

        if not item.get("rag_error"):
            return item
        last_item = item
        if attempt < _RAG_UNIT_FULL_BUILD_ATTEMPTS - 1:
            time.sleep(_RAG_UNIT_FULL_BUILD_RETRY_SLEEP_BASE * (attempt + 1))

    err = (last_item.get("rag_error") or "未知錯誤").strip()
    last_item["rag_error"] = f"{err}（已重試 {_RAG_UNIT_FULL_BUILD_ATTEMPTS} 次）"
    return last_item


def _rag_zip_build_counts(outputs: list[dict[str, Any]]) -> dict[str, int]:
    total = len(outputs)
    failed = sum(1 for o in outputs if o.get("rag_error"))
    return {"total": total, "built_ok": total - failed, "built_failed": failed}


def _persist_rag_build_metadata(body: PackRequest, pid: str, response: dict[str, Any]) -> None:
    """成功建置後更新 Rag 表並為每個成功輸出單元建立 Rag_Unit 記錄。"""
    supabase = get_supabase()
    ts = now_taipei_iso()

    # 先嘗試更新 Rag；即使失敗也要繼續寫 Rag_Unit（避免 schema cache 未同步時整段被跳過）。
    try:
        update_payload = {
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
            "updated_at": ts,
        }
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", body.rag_tab_id).eq("person_id", pid).execute()
    except Exception:
        pass

    outputs = response.get("outputs", [])
    for output in outputs:
        if output.get("rag_error"):
            continue
        ut = output.get("unit_type")
        try:
            unit_type_val = int(ut) if ut is not None else RAG_UNIT_TYPE_DEFAULT
        except (TypeError, ValueError):
            unit_type_val = RAG_UNIT_TYPE_DEFAULT
        if unit_type_val < 0 or unit_type_val > 4:
            unit_type_val = RAG_UNIT_TYPE_DEFAULT
        unit_transcription = ""
        text_fn = ""
        mp3_fn = ""
        yt_url = ""
        if unit_type_val in (RAG_UNIT_TYPE_TEXT, RAG_UNIT_TYPE_MP3, RAG_UNIT_TYPE_YOUTUBE):
            tp = (output.get("transcript_plain") or "").strip()
            if tp:
                unit_transcription = tp
            # text_file_name 僅 unit_type=2（文字單元）；勿與 User.user_type 混淆
            if unit_type_val == RAG_UNIT_TYPE_TEXT:
                text_fn = output.get("text_file_name") or ""
            if unit_type_val == RAG_UNIT_TYPE_MP3:
                mp3_fn = output.get("mp3_file_name") or ""
            if unit_type_val == RAG_UNIT_TYPE_YOUTUBE:
                yt_url = output.get("youtube_url") or ""
        unit_row = _rag_unit_default_row(
            body.rag_tab_id,
            pid,
            unit_name=output.get("unit_name", ""),
            unit_type=unit_type_val,
            repack_file_name=output.get("repack_filename", ""),
            rag_file_name=output.get("rag_filename", ""),
            rag_file_size=float(output.get("file_size") or 0),
            transcription=unit_transcription,
            text_file_name=text_fn,
            mp3_file_name=mp3_fn,
            youtube_url=yt_url,
        )
        try:
            supabase.table("Rag_Unit").insert(unit_row).execute()
        except Exception:
            pass


@router.get("/tabs", response_model=ListRagResponse)
def list_rag(
    request: Request,
    person_id: PersonId,
    local: bool | None = Query(
        None,
        description="僅回傳 Rag.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false",
    ),
):
    """
    列出 Rag 表內容（deleted=False），僅回傳與 query person_id 相符之列，Rag.local 須與 query local 相符（未傳 local 時依連線自動判定）。
    回傳列依 created_at 由舊到新排序。
    每筆 Rag 含 units（Rag_Unit 列表），每個 unit 含 quizzes（Rag_Quiz 列表）。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _rag_table_select(RAG_SELECT_ALL, exclude_deleted=True, local_match=local_filter)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        rag_tab_ids = list(dict.fromkeys(
            r.get("rag_tab_id") for r in data if r.get("rag_tab_id")
        ))
        units_by_tab = _units_by_rag_tab_ids(rag_tab_ids)

        all_unit_ids: list[int] = []
        for units in units_by_tab.values():
            for unit in units:
                uid = unit.get("rag_unit_id")
                if uid is not None:
                    try:
                        all_unit_ids.append(int(uid))
                    except (TypeError, ValueError):
                        pass
        all_unit_ids = list(dict.fromkeys(all_unit_ids))
        quizzes_by_unit = _quizzes_by_rag_unit_ids(all_unit_ids)

        for row in data:
            tab_id = row.get("rag_tab_id")
            units = units_by_tab.get(tab_id, []) if tab_id else []
            for unit in units:
                uid = unit.get("rag_unit_id")
                uid_int = int(uid) if uid is not None else None
                unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []
            row["units"] = units

        data = to_json_safe(data)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        logging.exception("GET /rag/tabs 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag 失敗: {e!s}")


@router.post("/tab/create")
def create_unit(body: CreateRagRequest, caller_person_id: PersonId):
    """
    只建立一筆 Rag 資料，接受 rag_tab_id、person_id、tab_name（必填）、local（選填，預設 false）。
    回傳新增的 rag_id、rag_tab_id、person_id、tab_name、local、created_at。
    """
    fid = (body.rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .insert(
                _rag_default_row(
                    fid,
                    tab_name=tab_name,
                    person_id=pid,
                    file_metadata=None,
                    local=body.local,
                )
            )
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=500, detail="新增 Rag 失敗")
        row = r.data[0]
        return {
            "rag_id": row["rag_id"],
            "rag_tab_id": row["rag_tab_id"],
            "tab_name": row.get("tab_name"),
            "person_id": row.get("person_id"),
            "local": row.get("local"),
            "created_at": to_taipei_iso(row.get("created_at")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tab/tab-name")
def update_unit_tab_name(body: UpdateRagUnitNameRequest, caller_person_id: PersonId):
    """
    更新既有 Rag 的 tab_name。以 rag_id（Rag 主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_id、rag_tab_id、person_id、tab_name、updated_at。
    """
    if body.rag_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 rag_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Rag")
            .select("rag_id, rag_tab_id, person_id")
            .eq("rag_id", body.rag_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_id 的 Rag 資料，或已刪除")
        row = sel.data[0]
        fid = row.get("rag_tab_id")
        pid = row.get("person_id")
        if ((pid or "").strip() != caller_person_id):
            raise HTTPException(status_code=403, detail="無權修改該 Rag")
        ts = now_taipei_iso()
        supabase.table("Rag").update({"tab_name": tab_name, "updated_at": ts}).eq("rag_id", body.rag_id).eq("deleted", False).execute()
        return {
            "rag_id": body.rag_id,
            "rag_tab_id": fid,
            "person_id": pid,
            "tab_name": tab_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tab/upload-zip")
async def upload_zip(
    caller_person_id: PersonId,
    file: UploadFile = File(...),
    rag_tab_id: str = Form(..., description="對應 tab/create 建立的 rag_tab_id，ZIP 會存於此路徑"),
    person_id: str = Form(..., description="寫入儲存路徑的 person_id，需與 tab/create 一致"),
):
    """
    Upload Zip：只做上傳並寫入資料庫。需先以 tab/create 建立該 rag_tab_id 的 Rag 資料。
    會更新該筆 Rag 的 file_metadata（filename、second_folders、file_size 等）與 file_size 欄位（皆為 MB）。
    回傳 file_metadata。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    fid = (rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            folders = get_second_level_folders_from_zip_file(zip_ref)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取 ZIP 檔案")

    resolved_person_id = (person_id or "").strip()
    if not resolved_person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if resolved_person_id != caller_person_id:
        raise HTTPException(status_code=400, detail="Form 的 person_id 與 query 不一致")

    try:
        supabase = get_supabase()
        r = supabase.table("Rag").select("rag_id, created_at").eq("rag_tab_id", fid).eq("person_id", resolved_person_id).execute()
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_tab_id 的 Rag 資料，請先呼叫 POST /rag/tab/create 建立")
        row = r.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        save_zip(
            contents,
            file.filename,
            folder=FOLDER_UPLOAD,
            person_id=resolved_person_id,
            tab_id=fid,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageApiError as e:
        if e.status == 413:
            raise HTTPException(
                status_code=413,
                detail=(
                    "ZIP 超過 Supabase Storage 允許的單檔大小上限；請縮小檔案，"
                    "或至 Supabase Dashboard → Project Settings → Storage 調高／確認方案限制。"
                ),
            ) from e
        raise HTTPException(status_code=502, detail=f"儲存上傳失敗: {e.message}") from e

    file_size_mb = _bytes_to_mb(len(contents))
    file_metadata = {
        "rag_id": row["rag_id"],
        "rag_tab_id": fid,
        "created_at": to_taipei_iso(row["created_at"]),
        "filename": file.filename,
        "second_folders": folders,
        "file_size": file_size_mb,
    }
    update_payload: dict[str, Any] = {
        "file_metadata": file_metadata,
        "file_size": file_size_mb,
        "updated_at": now_taipei_iso(),
    }
    try:
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", fid).eq("person_id", resolved_person_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return file_metadata


@router.post("/tab/build-rag-zip-stream", include_in_schema=False)
@router.post("/tab/build-rag-zip")
def build_rag_zip(
    body: PackRequest,
    caller_person_id: PersonId,
    repack_only: bool = Query(
        False,
        description="為 True 時強制不建 FAISS（unit_type=1 時 rag 改為 repack 複製）；不影響 unit_type=2/3/4 之逐字稿 rag ZIP",
    ),
):
    """
    依先前上傳的 ZIP（rag_tab_id）與 unit_list 重新打包。
    **FAISS 建置規則（逐 unit 判斷）**：`user_type==1`（且未強制關閉）且該 unit 之 `unit_type==1 (rag)` → 建 FAISS 並上傳至 rag；`unit_type` 為 2／3／4 時仍 repack 原 ZIP，但 **rag 區上傳內含單一 `transcript.md`（逐字稿全文）之 ZIP**，非 repack 複製；其餘 unit_type==0 等 → repack 同內容複製至 rag。
    可選 query **repack_only=true**：強制全部 unit 不建 FAISS；**不影響** 2／3／4 之逐字稿 rag ZIP 行為。
    可選 body **build_faiss**：`false` 同 repack_only；`true` 強制允許 FAISS（仍需 unit_type==1 觸發）；省略時依 user_type 判定。
    LLM API Key 僅在「最終會建 FAISS」（do_rag 為 True）時必填（依 person_id 自 User 表取得）。
    body.unit_types 為選填，與 unit_list 逗號分段對齊，寫入各 Rag_Unit.unit_type：0=未選、1=rag、2=文字、3=mp3、4=youtube。單元 2／3／4 成功時 **Rag_Unit.transcription** 寫入逐字稿全文；**text_file_name**（僅 **unit_type=2**，為上傳單元 ZIP 內該文字檔之檔名）／**mp3_file_name**（僅 3）／**youtube_url**（僅 4）寫入對應欄位。

    **回應為 NDJSON 串流**（`application/x-ndjson`），請以 `fetch` 讀取 `response.body`，勿使用單次 `response.json()`。
    每一輸出單元須 **成功上傳 repack**；rag 資料夾須 **成功寫入**（unit_type=1 且建 FAISS 為向量庫 ZIP；2／3／4 為逐字稿 md ZIP；其餘為 repack 同內容），且**上傳後能自儲存讀回非空檔**。
    整批成功時自動在 Rag_Unit 表建立對應記錄（每個輸出單元一筆）並更新 Rag.rag_metadata。
    整批任一有 `rag_error` 則 `complete.success` 為 false（不寫入 Rag 表，不建立 Rag_Unit）。

    事件列舉（每行一個物件）：
    - `{"type":"start","total":N,"source_rag_tab_id":"...","unit_list":"...","user_type":int,"build_faiss_request":bool|null,"repack_only":bool,"allow_faiss":bool}`（allow_faiss=各 unit 是否可建 FAISS，仍需 unit_type==1 才實際建）
    - `{"type":"building","index":i,"total":N,"completed_before":i-1,"filename":"..."}`
    - `{"type":"unit",...,"output":{...}}`：output 含 rag_mode（`faiss`＝向量庫；`transcript_md`＝逐字稿 md ZIP；`repack_copy`＝與 repack 同內容複製）、`transcript_plain`；**text_file_name** 僅 **unit_type=2** 有值（來源文字檔檔名）；**mp3_file_name** 僅 3；**youtube_url** 僅 4；rag_filename（物件鍵仍為 *_rag.zip）
    - `{"type":"complete","success":bool,"total","built_ok","built_failed","source_rag_tab_id","unit_list","outputs"}`

    串流階段 HTTP 狀態碼固定 **200**；請以最後一則 `type===complete` 的 `success` 判斷整批成敗。
    `POST /rag/tab/build-rag-zip-stream` 與本端點相同，僅自 OpenAPI 隱藏，供舊客戶端相容。
    """
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")

    path = _fetch_source_upload_zip_with_retries(pid, body.rag_tab_id)

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.unit_list)
    if not packed:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="unit_list 為空或格式錯誤，例：220222+220301")

    api_key, user_type_val = _fetch_user_llm_key_and_user_type(pid)

    # 是否「允許建 FAISS」：user_type==1 且未強制關閉
    # 即使允許，每個 unit 仍需 unit_type==1 才實際建 FAISS（見下方 _do_rag_for_unit）
    if repack_only or body.build_faiss is False:
        allow_faiss = False
    elif body.build_faiss is True:
        allow_faiss = True
    else:
        allow_faiss = (user_type_val == 1)

    # api_key 只在真正會建 FAISS 時才需要；若 allow_faiss 但 key 缺失則提早報錯
    if allow_faiss and not api_key:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail="該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
        )

    total = len(packed)
    unit_types_per_task = _unit_types_per_task(body.unit_types, total)

    def _do_rag_for_unit(ut: int) -> bool:
        """只有 allow_faiss 且 unit_type==1 (rag) 時才建 FAISS；其餘走 repack 分支（2/3/4 時 rag 為逐字稿 ZIP）。"""
        return allow_faiss and (ut == RAG_UNIT_TYPE_RAG)

    def ndjson_events():
        outputs: list[dict[str, Any]] = []
        try:
            yield (
                json.dumps(
                    {
                        "type": "start",
                        "total": total,
                        "source_rag_tab_id": body.rag_tab_id,
                        "unit_list": body.unit_list,
                        "user_type": user_type_val,
                        "build_faiss_request": body.build_faiss,
                        "repack_only": repack_only,
                        "allow_faiss": allow_faiss,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for idx, (zip_bytes, filename) in enumerate(packed):
                yield (
                    json.dumps(
                        {
                            "type": "building",
                            "index": idx + 1,
                            "total": total,
                            "completed_before": idx,
                            "filename": filename,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ut = unit_types_per_task[idx]
                item = _build_one_rag_zip_output_item(
                    body,
                    pid,
                    api_key or "",
                    zip_bytes,
                    filename,
                    do_rag=_do_rag_for_unit(ut),
                    unit_type=ut,
                )
                outputs.append(item)
                yield (
                    json.dumps(
                        {"type": "unit", "index": idx + 1, "total": total, "output": item},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            success = not any(o.get("rag_error") for o in outputs)
            counts = _rag_zip_build_counts(outputs)
            response = {
                "source_rag_tab_id": body.rag_tab_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if success:
                _persist_rag_build_metadata(body, pid, response)
            complete_ev: dict[str, Any] = {
                "type": "complete",
                "success": success,
                "source_rag_tab_id": body.rag_tab_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if not success:
                complete_ev["message"] = "RAG ZIP 建立失敗（請修正後重試）"
            yield json.dumps(complete_ev, ensure_ascii=False) + "\n"
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(
        ndjson_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


def _do_delete_rag_file_by_tab_id(fid: str) -> tuple[bool, str]:
    """依 rag_tab_id 將 Rag 未刪除列軟刪除，同時軟刪除對應 Rag_Unit，並刪除 storage 資料夾。"""
    supabase = get_supabase()
    sel = supabase.table("Rag").select("person_id").eq("rag_tab_id", fid).eq("deleted", False).execute()
    if not sel.data:
        raise HTTPException(status_code=404, detail="找不到該 rag_tab_id 的 Rag 資料，或已刪除")
    pids_ordered: list[str] = []
    seen: set[str] = set()
    for row in sel.data:
        pid = (row.get("person_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        pids_ordered.append(pid)
    primary_pid = pids_ordered[0] if pids_ordered else ""
    try:
        supabase.table("Rag").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("rag_tab_id", fid).eq("deleted", False).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Rag 表失敗: {e}")
    try:
        supabase.table("Rag_Unit").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("rag_tab_id", fid).eq("deleted", False).execute()
    except Exception:
        pass
    folder_deleted = False
    for pid in pids_ordered:
        try:
            if delete_tab_folder(pid, fid):
                folder_deleted = True
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return folder_deleted, primary_pid


@router.post("/tab/delete/{rag_tab_id}", status_code=200)
def delete_rag_file(
    _person_id: PersonId,
    rag_tab_id: str = PathParam(..., description="要刪除的 rag_tab_id"),
):
    """
    POST /rag/tab/delete/{rag_tab_id}。
    軟刪除：將 Rag 表該 rag_tab_id 之未刪除列 deleted 設為 true，同時軟刪除所有對應 Rag_Unit，並刪除 storage 資料夾。
    """
    fid = (rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    folder_deleted, pid = _do_delete_rag_file_by_tab_id(fid)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "rag_tab_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }


@router.get("/tab/units")
def list_rag_units(
    _caller_person_id: PersonId,
    rag_tab_id: str = Query(..., description="要列出 Rag_Unit 的 rag_tab_id"),
):
    """
    依 rag_tab_id 列出所有未刪除的 Rag_Unit，每個 unit 含關聯的 Rag_Quiz（quizzes）。
    依 created_at 由舊到新排序。
    """
    try:
        fid = (rag_tab_id or "").strip()
        if not fid:
            raise HTTPException(status_code=400, detail="請傳入 rag_tab_id")
        supabase = get_supabase()
        units_resp = (
            supabase.table("Rag_Unit")
            .select("*")
            .eq("rag_tab_id", fid)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
        units = units_resp.data or []

        unit_ids: list[int] = []
        for u in units:
            uid = u.get("rag_unit_id")
            if uid is not None:
                try:
                    unit_ids.append(int(uid))
                except (TypeError, ValueError):
                    pass
        unit_ids = list(dict.fromkeys(unit_ids))
        quizzes_by_unit = _quizzes_by_rag_unit_ids(unit_ids)

        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        units = to_json_safe(units)
        return {"units": units, "count": len(units)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("GET /rag/units 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag_Unit 失敗: {e!s}")


@router.put("/tab/unit/unit-name")
def update_rag_unit_name(body: UpdateRagUnitUnitNameRequest, caller_person_id: PersonId):
    """
    更新既有 Rag_Unit 的 unit_name。以 rag_unit_id（主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_unit_id、rag_tab_id、person_id、unit_name、updated_at。
    """
    if body.rag_unit_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 rag_unit_id")
    unit_name = (body.unit_name or "").strip()
    if not unit_name:
        raise HTTPException(status_code=400, detail="請傳入 unit_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id, rag_tab_id, person_id")
            .eq("rag_unit_id", body.rag_unit_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_unit_id 的 Rag_Unit 資料，或已刪除")
        row = sel.data[0]
        pid = row.get("person_id")
        if (pid or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Rag_Unit")
        ts = now_taipei_iso()
        supabase.table("Rag_Unit").update({"unit_name": unit_name, "updated_at": ts}).eq("rag_unit_id", body.rag_unit_id).eq("deleted", False).execute()
        return {
            "rag_unit_id": body.rag_unit_id,
            "rag_tab_id": row.get("rag_tab_id"),
            "person_id": pid,
            "unit_name": unit_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tab/unit/quiz/create", summary="Rag Create Quiz (no LLM)", operation_id="rag_create_quiz")
def insert_rag_quiz_row(body: InsertRagQuizRowRequest, caller_person_id: PersonId):
    """
    依 `rag_tab_id`／`rag_unit_id` 解析 Rag_Unit 後新增一筆空白 `Rag_Quiz`，**不呼叫 LLM**。`rag_quiz_id` 由資料庫自動產生並於回傳中帶出。
    LLM 出題請用 `POST /rag/tab/unit/quiz/llm-generate`。
    """
    try:
        supabase = get_supabase()
        req_tab = (body.rag_tab_id or "").strip()
        resolved_unit_id = int(body.rag_unit_id or 0)

        u: dict[str, Any] | None = None

        if resolved_unit_id > 0:
            sel = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id, rag_tab_id, person_id, unit_name")
                .eq("rag_unit_id", resolved_unit_id)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
            if sel.data:
                u = sel.data[0]
        else:
            if not req_tab:
                raise HTTPException(
                    status_code=400,
                    detail="請傳入 rag_unit_id（>0），或傳入 rag_tab_id 且該 tab 下僅有一筆 Rag_Unit",
                )
            sel = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id, rag_tab_id, person_id, unit_name")
                .eq("rag_tab_id", req_tab)
                .eq("deleted", False)
                .execute()
            )
            rows = sel.data or []
            if len(rows) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"rag_tab_id 需唯一對應一筆 Rag_Unit（目前 {len(rows)} 筆），請改傳 rag_unit_id",
                )
            u = rows[0]

        if u is None:
            raise HTTPException(status_code=404, detail="找不到該 rag_unit_id 的 Rag_Unit 資料，或已刪除")

        uid = int(u.get("rag_unit_id") or 0)
        if req_tab and (u.get("rag_tab_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_tab_id 與 rag_unit_id 對應之 Rag_Unit 不一致")
        pid = (u.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權於該 Rag_Unit 新增題目")
        rag_tab_id = (u.get("rag_tab_id") or "").strip()
        if not rag_tab_id:
            raise HTTPException(status_code=400, detail="該 Rag_Unit 的 rag_tab_id 為空，無法寫入 Rag_Quiz")
        quiz_name = (u.get("unit_name") or "").strip()
        ts = now_taipei_iso()
        quiz_row: dict[str, Any] = {
            "rag_tab_id": rag_tab_id,
            "rag_unit_id": uid,
            "person_id": pid,
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": "",
            "quiz_content": "",
            "quiz_hint": "",
            "quiz_answer_reference": "",
            "answer_user_prompt_text": "",
            "answer_content": "",
            "answer_critique": None,
            "for_exam": False,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = supabase.table("Rag_Quiz").insert(quiz_row).execute()
        if not ins.data or len(ins.data) == 0:
            raise HTTPException(status_code=500, detail="寫入 Rag_Quiz 失敗（無回傳資料）")
        row = ins.data[0]
        return to_json_safe(
            {
                "rag_quiz_id": row.get("rag_quiz_id"),
                "rag_tab_id": row.get("rag_tab_id"),
                "rag_unit_id": row.get("rag_unit_id"),
                "person_id": row.get("person_id"),
                "quiz_name": row.get("quiz_name"),
                "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
                "quiz_content": row.get("quiz_content"),
                "quiz_hint": row.get("quiz_hint"),
                "quiz_answer_reference": row.get("quiz_answer_reference"),
                "answer_user_prompt_text": row.get("answer_user_prompt_text"),
                "answer_content": row.get("answer_content"),
                "answer_grade": _quiz_grade_from_answer_critique(row.get("answer_critique")),
                "answer_critique": row.get("answer_critique"),
                "for_exam": row.get("for_exam"),
                "deleted": row.get("deleted"),
                "updated_at": row.get("updated_at"),
                "created_at": row.get("created_at"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("POST /rag/tab/unit/quiz/create 錯誤")
        raise HTTPException(status_code=500, detail=str(e))

