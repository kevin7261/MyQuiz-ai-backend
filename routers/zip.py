"""
ZIP 與 RAG 相關 API 模組。路徑層級與排序與 Exam 對齊（見 utils.openapi_order、README API 目錄）。

**分頁**：GET /rag/pages → GET /rag/page/units → POST /rag/page/create → POST /rag/page/create-upload-zip
→ PUT /rag/page/tab-name → PUT /rag/page/delete/{rag_page_id} → POST /rag/page/upload-zip → POST /rag/page/build-rag-zip（-stream 別名）

**單元**：PUT /rag/page/unit/unit-name → GET /rag/page/unit/mp3-file → GET /rag/page/unit/youtube-url

**題目**：POST /rag/page/unit/quiz/create → PUT /rag/page/unit/quiz/quiz-name → PUT /rag/page/unit/quiz/delete/{rag_quiz_id}
→（followup／for-exam／llm-* 見 routers/grade.py）

**舊路徑**：GET /rag/unit/text、/rag/unit/mp3-file、/rag/unit/youtube-url

GET /rag/pages 須 course_id、person_id；`local` 未傳時依連線判定。build-rag-zip 見路由 docstring。
"""

import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from postgrest.exceptions import APIError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from storage3.exceptions import StorageApiError

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.llm_key import get_rag_api_key
from utils.openapi import openapi_body
from utils.serialization import to_json_safe
from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
    repack_zip_stem_from_filename,
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
from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.supabase import get_supabase
from utils.db_schema import (
    ACTIVE_DELETED_FILTER,
    RAG_COURSE_ID_DEFAULT,
    RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP,
    RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST,
    RAG_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST,
    USER_COURSE_RELATION_TABLE,
    rag_quiz_list_row,
)
from utils.rag_course import (
    execute_with_course_id_fallback,
    insert_rag_child_row,
    require_rag_tab_owner,
    resolve_rag_tab_owner_person_id,
    select_without_course_id_if_needed,
)
from utils.rag_stem import transcript_from_row
from utils.rag_exam_setting import is_localhost_request
from utils.media import audio_media_type_for_suffix
from utils.rag_transcript import (
    build_transcript_md_zip_bytes,
    extract_transcript_for_rag_build,
    infer_unit_type_when_unspecified,
    pick_audio_from_upload_zip_with_folder_fallback,
    read_repack_zip_bytes,
    read_upload_zip_bytes,
)
router = APIRouter(prefix="/rag", tags=["rag"])

_logger = logging.getLogger(__name__)

RAG_SELECT_ALL = "*"

BYTES_PER_MB = 1024 * 1024

# Rag_Unit.unit_type（PostgreSQL smallint）：0=未選、1=rag、2=文字、3=mp3、4=youtube
RAG_UNIT_TYPE_DEFAULT = 0
RAG_UNIT_TYPE_RAG = 1
RAG_UNIT_TYPE_TEXT = 2
RAG_UNIT_TYPE_MP3 = 3
RAG_UNIT_TYPE_YOUTUBE = 4


def _bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / BYTES_PER_MB


def _ndjson_line(obj: dict[str, Any]) -> str:
    """序列化單一 NDJSON 事件（UTF-8 原樣，不轉 ASCII），結尾附換行。"""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _safe_unlink(p: Path) -> None:
    """刪除暫存檔；忽略檔案不存在或刪除失敗。"""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _fetch_user_type(person_id: str, course_id: int) -> int:
    """依 person_id、course_id 自 User_Course_Relation 取得 user_type；無列時回傳 0。"""
    pid = (person_id or "").strip()
    if not pid:
        return 0
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_COURSE_RELATION_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .or_(ACTIVE_DELETED_FILTER)
            .order("course_user_id")
            .limit(1)
            .execute()
        )
        if not resp.data:
            return 0
        ut = resp.data[0].get("user_type")
        try:
            return int(ut) if ut is not None else 0
        except (TypeError, ValueError):
            return 0
    except Exception:
        return 0


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


def _split_csv_optional_ints(csv: str, task_count: int) -> list[int | None]:
    """與 packed 任務數對齊；逗號分段，該段空白或無法解析為整數則為 None（呼叫端改用預設）。"""
    parts = [p.strip() for p in csv.split(",")] if (csv or "").strip() else []
    out: list[int | None] = []
    for i in range(task_count):
        if i < len(parts) and parts[i]:
            try:
                out.append(int(parts[i]))
            except ValueError:
                out.append(None)
        else:
            out.append(None)
    return out


def _clamp_chunk_pair(size: int, overlap: int) -> tuple[int, int]:
    """chunk 用於向量分段；夾在合理範圍並確保 overlap < size。"""
    size = max(64, min(int(size), 32000))
    overlap = max(0, min(int(overlap), max(0, size - 1)))
    return size, overlap


def _unit_name_overrides_per_task(unit_names: str | list[str] | None, task_count: int) -> list[str | None]:
    """與 packed 任務同序；非空字串表示覆寫 Rag_Unit.unit_name（顯示名），None 表示與 folder_combination（stem）相同。"""
    if unit_names is None or unit_names == "":
        return [None] * task_count
    if isinstance(unit_names, list):
        parts = [("" if x is None else str(x)).strip() for x in unit_names]
    else:
        s = str(unit_names).strip()
        parts = [p.strip() for p in s.split(",")] if s else []
    out: list[str | None] = []
    for i in range(task_count):
        if i < len(parts) and parts[i]:
            out.append(parts[i])
        else:
            out.append(None)
    return out


def _chunk_params_per_task(
    sizes_csv: str,
    overlaps_csv: str,
    task_count: int,
    default_size: int,
    default_overlap: int,
) -> list[tuple[int, int]]:
    """每個 repack 任務一組 (rag_chunk_size, rag_chunk_overlap)；CSV 缺段則用 default_*（經 clamp）。"""
    sz = _split_csv_optional_ints(sizes_csv, task_count)
    ov = _split_csv_optional_ints(overlaps_csv, task_count)
    d_sz, d_ov = _clamp_chunk_pair(default_size, default_overlap)
    pairs: list[tuple[int, int]] = []
    for i in range(task_count):
        raw_s = sz[i] if i < len(sz) else None
        raw_o = ov[i] if i < len(ov) else None
        s = raw_s if raw_s is not None else d_sz
        o = raw_o if raw_o is not None else d_ov
        pairs.append(_clamp_chunk_pair(s, o))
    return pairs


def _rag_default_row(
    rag_page_id: str,
    *,
    tab_name: str | None = None,
    person_id: str | None = None,
    course_id: int = RAG_COURSE_ID_DEFAULT,
    file_metadata: Any = None,
    local: bool = False,
) -> dict[str, Any]:
    """Rag 表一筆新增時的預設欄位；鍵順序同 public.Rag（rag_page_id→person_id→course_id→…；不含 rag_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    row: dict[str, Any] = {
        "rag_page_id": rag_page_id,
        "person_id": person_id if person_id is not None else "",
        "course_id": course_id,
        "tab_name": tab_name if tab_name is not None else "",
        "file_metadata": file_metadata,
    }
    # rag_metadata／rag_chunk_size 等欄位在 Rag_Unit 層管理，建立時不主動寫入以避免 schema 未同步時 500
    row["local"] = local
    row["deleted"] = False
    row["created_at"] = ts
    row["updated_at"] = ts
    return row


def _rag_unit_default_row(
    rag_page_id: str,
    person_id: str,
    *,
    course_id: int = RAG_COURSE_ID_DEFAULT,
    unit_name: str = "",
    folder_combination: str = "",
    unit_type: int = RAG_UNIT_TYPE_RAG,
    repack_file_name: str = "",
    rag_file_name: str = "",
    rag_file_size: float = 0.0,
    transcript: str = "",
    text_file_name: str = "",
    mp3_file_name: str = "",
    youtube_url: str = "",
    rag_chunk_size: int = 1000,
    rag_chunk_overlap: int = 200,
) -> dict[str, Any]:
    """Rag_Unit 表一筆新增時的預設欄位（含 rag_chunk_size／rag_chunk_overlap，與 build-rag-zip 向量分段一致）。"""
    ts = now_taipei_iso()
    return {
        "rag_page_id": rag_page_id,
        "person_id": person_id,
        "course_id": course_id,
        "unit_name": unit_name,
        "folder_combination": folder_combination,
        "unit_type": unit_type,
        "rag_chunk_size": int(rag_chunk_size),
        "rag_chunk_overlap": int(rag_chunk_overlap),
        "repack_file_name": repack_file_name,
        "rag_file_name": rag_file_name,
        "rag_file_size": rag_file_size,
        "transcript": transcript,
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
    course_id: int | None = None,
) -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。exclude_deleted=True 時僅回傳 deleted=False。local_match 若指定則僅回傳 Rag.local 與其相符的列。course_id 若指定則僅回傳該課程。依 created_at 升序（舊→新）。"""
    supabase = get_supabase()
    q = supabase.table("Rag").select(select_columns)
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    if course_id is not None:
        q = q.eq("course_id", course_id)
    q = q.order("created_at", desc=False)
    resp = q.execute()
    return resp.data or []


def _units_by_rag_page_ids(
    rag_page_ids: list[str],
    *,
    course_id: int | None = None,
) -> dict[str, list[dict]]:
    """依 rag_page_id 查詢 Rag_Unit 表，回傳 rag_page_id -> list of unit 列。僅回傳 deleted=False。course_id 若指定則僅回傳該課程。依 created_at 升序。"""
    if not rag_page_ids:
        return {}
    supabase = get_supabase()

    def build_unit_query(with_course_filter: bool):
        q = (
            supabase.table("Rag_Unit")
            .select("*")
            .in_("rag_page_id", rag_page_ids)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("created_at", desc=False)

    resp = execute_with_course_id_fallback("Rag_Unit", build_unit_query, course_id)
    rows = resp.data or []
    out: dict[str, list[dict]] = {tid: [] for tid in rag_page_ids}
    for row in rows:
        tid = row.get("rag_page_id")
        if tid is not None:
            out.setdefault(tid, []).append(row)
    return out


def _quizzes_by_rag_unit_ids(
    rag_unit_ids: list[int],
    *,
    course_id: int | None = None,
) -> dict[int, list[dict]]:
    """依 rag_unit_id 查詢 Rag_Quiz，回傳 rag_unit_id -> quizzes[]（含 follow_up）。僅 deleted=False。"""
    if not rag_unit_ids:
        return {}
    supabase = get_supabase()

    def build_quiz_query(with_course_filter: bool, *, columns: str):
        cols = select_without_course_id_if_needed("Rag_Quiz", columns, with_course_filter)
        q = (
            supabase.table("Rag_Quiz")
            .select(cols)
            .in_("rag_unit_id", rag_unit_ids)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("created_at", desc=False)

    try:
        resp = execute_with_course_id_fallback(
            "Rag_Quiz",
            lambda with_course: build_quiz_query(with_course, columns="*"),
            course_id,
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "quiz_history_list" in msg:
            try:
                resp = execute_with_course_id_fallback(
                    "Rag_Quiz",
                    lambda with_course: build_quiz_query(
                        with_course, columns=RAG_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST
                    ),
                    course_id,
                )
            except APIError as e2:
                msg2 = (e2.message or "").lower()
                if e2.code == "42703" and "follow_up" in msg2:
                    resp = execute_with_course_id_fallback(
                        "Rag_Quiz",
                        lambda with_course: build_quiz_query(
                            with_course,
                            columns=RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST,
                        ),
                        course_id,
                    )
                else:
                    raise
        elif e.code == "42703" and "follow_up" in msg:
            resp = execute_with_course_id_fallback(
                "Rag_Quiz",
                lambda with_course: build_quiz_query(
                    with_course, columns=RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP
                ),
                course_id,
            )
        else:
            raise
    rows = resp.data or []
    out: dict[int, list[dict]] = {uid: [] for uid in rag_unit_ids}
    for row in rows:
        uid = row.get("rag_unit_id")
        if uid is not None:
            try:
                out.setdefault(int(uid), []).append(rag_quiz_list_row(row))
            except (TypeError, ValueError):
                pass
    return out


class ListRagResponse(BaseModel):
    """GET /rag/pages 回應：每筆 Rag 含 units（Rag_Unit），每個 unit 含 quizzes（Rag_Quiz，含 follow_up、quiz_history_list）；皆已依 query course_id 篩選。"""
    rags: list[dict]
    count: int


class RagUnitMp3FileResponse(BaseModel):
    """GET /rag/page/unit/mp3-file 回應。"""
    rag_unit_id: int
    rag_page_id: str
    audio_base64: str
    media_type: str
    filename: str
    transcript: str = ""


class RagUnitYoutubeUrlResponse(BaseModel):
    """GET /rag/page/unit/youtube-url 回應。"""
    rag_unit_id: int
    rag_page_id: str
    youtube_url: str
    transcript: str = ""


class CreateRagRequest(BaseModel):
    """POST /rag/page/create：欄位順序同 public.Rag（rag_page_id, person_id, tab_name, local；不含 rag_id／course_id／deleted／時間戳）。"""
    rag_page_id: str = Field(..., description="Rag 的 tab 識別，對應 Rag 表 rag_page_id 欄位")
    person_id: str = Field(..., description="使用者/路徑識別")
    tab_name: str = Field(..., description="Rag 顯示名稱，寫入 Rag 表 tab_name 欄位")
    local: bool = Field(False, description="是否為本機 RAG，寫入 Rag 表 local 欄位")


class UpdateRagUnitNameRequest(BaseModel):
    """PUT /rag/page/tab-name：請求僅含 rag_id（主鍵）、tab_name；勿傳 rag_page_id。"""
    rag_id: int = Field(..., description="Rag 表主鍵（整數）；辨識請用 rag_id，非 rag_page_id")
    tab_name: str = Field(..., description="新的顯示名稱，寫入 Rag 表 tab_name 欄位")


class PackRequest(BaseModel):
    """
    rag_page_id、person_id（同 public.Rag）→ unit_list → Rag_Unit 相關欄（unit_name、unit_type、transcript、rag_chunk_*）。
    rag_chunk_size／rag_chunk_overlap：全批預設（寫入 Rag_Unit、建 FAISS 時用）。
    rag_chunk_sizes／rag_chunk_overlaps：可選逗號字串或整數陣列（JSON），與 unit_list 解出之任務數同序；某段空白則該段用 rag_chunk_size／rag_chunk_overlap。
    unit_names：可選逗號字串或字串陣列（JSON），與任務同序；某段 strip 後非空則覆寫該單元 Rag_Unit.unit_name（顯示名），空白則與 folder_combination 相同（皆為檔名 stem）。
    """
    rag_page_id: str
    person_id: str
    unit_list: str  # 指定要打包的資料夾；例："220222+220301"（加號=同一 ZIP 多資料夾）；結果存入 Rag_Unit 表
    unit_names: str | list[str] | None = Field(
        default="",
        description="可選；逗號字串或 JSON 字串陣列，與 packed 任務同序；該段 strip 後非空則覆寫 Rag_Unit.unit_name（顯示名），空段則與 folder_combination 相同（檔名 stem）",
    )
    unit_types: str = ""
    transcripts: list[str] | None = Field(
        default=None,
        description="可選；與 unit_list 逗號分段對齊；非空時覆寫逐字稿全文",
    )
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_chunk_sizes: str = Field(
        "",
        description="可選；逗號字串或 [1000,800] 陣列，與 packed 任務同序；空段→該任務用 rag_chunk_size",
    )
    rag_chunk_overlaps: str = Field(
        "",
        description="可選；逗號字串或 [200,100] 陣列，與 packed 任務同序；空段→該任務用 rag_chunk_overlap",
    )

    @field_validator("rag_chunk_sizes", "rag_chunk_overlaps", mode="before")
    @classmethod
    def _coerce_chunk_segments_csv(cls, v: Any) -> str:
        """相容前端傳 JSON 陣列；統一成逗號字串供 _chunk_params_per_task 解析。"""
        if v is None:
            return ""
        if isinstance(v, list):
            parts: list[str] = []
            for x in v:
                try:
                    parts.append(str(int(x)))
                except (TypeError, ValueError):
                    parts.append("")
            return ",".join(parts)
        if isinstance(v, str):
            return v.strip()
        return str(v)

    build_faiss: bool | None = Field(
        default=None,
        description="省略時依 User_Course_Relation.user_type；False 時僅複製 repack 至 rag；True 時強制建向量 RAG ZIP",
    )

    @field_validator("unit_names", mode="before")
    @classmethod
    def _coerce_unit_names(cls, v: Any) -> str | list[str]:
        if v is None:
            return ""
        if isinstance(v, list):
            return [("" if x is None else str(x)) for x in v]
        if isinstance(v, str):
            return v.strip()
        return str(v)


class UpdateRagUnitUnitNameRequest(BaseModel):
    """PUT /rag/page/unit/unit-name：更新 Rag_Unit 的 unit_name。"""
    rag_unit_id: int = Field(..., description="Rag_Unit 表主鍵")
    unit_name: str = Field(..., description="新的 unit_name")


class InsertRagQuizRowRequest(BaseModel):
    """
    POST /rag/page/unit/quiz/create：欄位順序對齊 public.Rag_Quiz 之關聯欄（rag_page_id、rag_unit_id）。
    `rag_page_id` 與 `rag_unit_id` 二擇一定位 Rag_Unit：
    - `rag_unit_id > 0`：以主鍵載入；若同傳 `rag_page_id`（非空）則須與該列一致。
    - `rag_unit_id == 0`：`rag_page_id`（非空）須在該名下**唯一**一筆未刪除之 Rag_Unit，否則 400。
    """

    rag_page_id: str = Field("", description="Rag tab 識別；與 rag_unit_id 併用見上")
    rag_unit_id: int = Field(0, ge=0, description="Rag_Unit 主鍵；0 表示改由 rag_page_id 唯一解析")


class UpdateRagQuizQuizNameRequest(BaseModel):
    """PUT /rag/page/unit/quiz/quiz-name：更新 Rag_Quiz 的 quiz_name。"""
    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 表主鍵")
    quiz_name: str = Field(..., description="新的 quiz_name")


_MIN_RAG_ZIP_BYTES = 22
_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS = 80
_SOURCE_UPLOAD_ZIP_SLEEP_SEC = 0.45
_RAG_ZIP_VERIFY_MAX_ATTEMPTS = 18
_RAG_ZIP_VERIFY_SLEEP_INITIAL = 0.35
_RAG_ZIP_VERIFY_SLEEP_MAX = 3.0
_RAG_UNIT_FULL_BUILD_ATTEMPTS = 3
_RAG_UNIT_FULL_BUILD_RETRY_SLEEP_BASE = 0.65


def _try_read_upload_zip_once(pid: str, rag_page_id: str) -> Path | None:
    path = get_zip_path(rag_page_id) or get_zip_path_by_person(pid, rag_page_id)
    if not path or not path.exists():
        return None
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.namelist()
        return path
    except zipfile.BadZipFile:
        _safe_unlink(path)
        return None
    except Exception:
        _safe_unlink(path)
        return None


def _fetch_source_upload_zip_with_retries(pid: str, rag_page_id: str) -> Path:
    last_fail = "無法取得路徑或下載失敗"
    for attempt in range(_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS):
        path = _try_read_upload_zip_once(pid, rag_page_id)
        if path is not None:
            return path
        last_fail = f"第 {attempt + 1}/{_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS} 次仍無法讀取上傳 ZIP"
        time.sleep(_SOURCE_UPLOAD_ZIP_SLEEP_SEC)
    raise HTTPException(
        status_code=503,
        detail=(
            f"多次重試後仍無法讀取上傳的 ZIP（rag_page_id={rag_page_id}）。"
            f"若剛上傳完請稍後再試；{last_fail}"
        ),
    )


def _verify_saved_rag_zip_readable(rag_zip_page_id: str) -> str | None:
    delay = _RAG_ZIP_VERIFY_SLEEP_INITIAL
    last_err = "RAG ZIP 上傳後無法從儲存讀回驗證"
    for _ in range(_RAG_ZIP_VERIFY_MAX_ATTEMPTS):
        verify_path = get_zip_path(rag_zip_page_id)
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
            _safe_unlink(verify_path)
        time.sleep(delay)
        delay = min(delay * 1.3, _RAG_ZIP_VERIFY_SLEEP_MAX)
    return last_err


def _build_repack_or_transcript_rag_item(
    body: PackRequest,
    pid: str,
    zip_bytes: bytes,
    filename: str,
    repack_page_id: str,
    *,
    unit_type: int,
    task_rag_chunk_size: int,
    task_rag_chunk_overlap: int,
    transcript_override: str | None,
) -> dict[str, Any]:
    """
    do_rag=False 分支（不建 FAISS）：repack 照舊上傳。
    unit_type 為 2/3/4：rag 區上傳逐字稿之 transcript.md ZIP（rag_mode=transcript_md），output 含 transcript_plain 與對應欄位。
    若 unit_type==0（請求漏傳對齊的 unit_types），會由單元 ZIP 推斷：恰一音訊＋一文字檔→3；否則恰一文字檔→2（完整保留 .md 原文）。YouTube（4）須明確傳。
    無法推斷時：將與 repack 相同內容複製至 rag（rag_mode=repack_copy），transcript 為空。
    """
    effective_ut = infer_unit_type_when_unspecified(unit_type, zip_bytes)
    item_zip: dict[str, Any] = {
        "filename": filename,
        "folder_combination": repack_page_id,
        "unit_name": repack_page_id,
        "repack_filename": f"{repack_page_id}.zip",
        "rag_filename": "",
        "unit_type": effective_ut,
        "rag_mode": "repack_copy",
        "transcript_plain": "",
        "text_file_name": "",
        "mp3_file_name": "",
        "youtube_url": "",
        "rag_chunk_size": int(task_rag_chunk_size),
        "rag_chunk_overlap": int(task_rag_chunk_overlap),
    }
    if effective_ut != unit_type:
        item_zip["unit_type_declared"] = unit_type
    try:
        page_id = save_zip(
            zip_bytes,
            filename,
            folder=FOLDER_REPACK,
            person_id=pid,
            parent_page_id=body.rag_page_id,
            page_id=repack_page_id,
        )
        item_zip["repack_filename"] = f"{page_id}.zip"
        rag_zip_page_id = f"{page_id}_rag"
        item_zip["rag_filename"] = f"{rag_zip_page_id}.zip"
        rag_payload: bytes | None = None
        if effective_ut in (RAG_UNIT_TYPE_TEXT, RAG_UNIT_TYPE_MP3, RAG_UNIT_TYPE_YOUTUBE):
            item_zip["rag_mode"] = "transcript_md"
            try:
                extracted = extract_transcript_for_rag_build(zip_bytes, effective_ut)
            except ValueError as e:
                item_zip["rag_error"] = str(e)
            else:
                item_zip["transcript_plain"] = extracted.get("transcript") or ""
                item_zip["text_file_name"] = extracted.get("text_file_name") or ""
                item_zip["mp3_file_name"] = extracted.get("mp3_file_name") or ""
                item_zip["youtube_url"] = extracted.get("youtube_url") or ""
                if transcript_override is not None:
                    ov = (
                        transcript_override
                        if isinstance(transcript_override, str)
                        else str(transcript_override)
                    )
                    if ov.strip() != "":
                        item_zip["transcript_plain"] = ov
                rag_payload = build_transcript_md_zip_bytes(item_zip["transcript_plain"])
                save_zip(
                    rag_payload,
                    f"{page_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_page_id=body.rag_page_id,
                    page_id=rag_zip_page_id,
                )
        else:
            rag_payload = zip_bytes
            save_zip(
                zip_bytes,
                f"{page_id}.zip",
                folder=FOLDER_RAG,
                person_id=pid,
                parent_page_id=body.rag_page_id,
                page_id=rag_zip_page_id,
            )
        if not item_zip.get("rag_error"):
            verify_err = _verify_saved_rag_zip_readable(rag_zip_page_id)
            if verify_err:
                item_zip["rag_error"] = verify_err
        item_zip["file_size"] = _bytes_to_mb(
            len(rag_payload) if rag_payload is not None else len(zip_bytes)
        )
    except Exception as e:
        item_zip["rag_error"] = str(e)
    return item_zip


def _build_faiss_rag_item(
    body: PackRequest,
    pid: str,
    api_key: str,
    zip_bytes: bytes,
    filename: str,
    repack_page_id: str,
    *,
    unit_type: int,
    task_rag_chunk_size: int,
    task_rag_chunk_overlap: int,
) -> dict[str, Any]:
    """do_rag=True 分支：repack 上傳後以 FAISS 建向量庫 RAG ZIP 上傳至 rag，失敗時重試。"""
    last_item: dict[str, Any] = {}
    for attempt in range(_RAG_UNIT_FULL_BUILD_ATTEMPTS):
        item: dict[str, Any] = {
            "filename": filename,
            "folder_combination": repack_page_id,
            "unit_name": repack_page_id,
            "repack_filename": f"{repack_page_id}.zip",
            "rag_filename": f"{repack_page_id}_rag.zip",
            "unit_type": unit_type,
            "rag_mode": "faiss",
            "rag_chunk_size": int(task_rag_chunk_size),
            "rag_chunk_overlap": int(task_rag_chunk_overlap),
        }
        rag_bytes_out: bytes | None = None
        try:
            from utils.rag_faiss import make_rag_zip_from_zip_path

            page_id = save_zip(
                zip_bytes,
                filename,
                folder=FOLDER_REPACK,
                person_id=pid,
                parent_page_id=body.rag_page_id,
                page_id=repack_page_id,
            )
            rag_zip_page_id = f"{page_id}_rag"
            item["repack_filename"] = f"{page_id}.zip"
            item["rag_filename"] = f"{rag_zip_page_id}.zip"

            fd, repack_tmp = tempfile.mkstemp(suffix=".zip", prefix="myquizai_repack_")
            os.close(fd)
            repack_local = Path(repack_tmp)
            try:
                repack_local.write_bytes(zip_bytes)
                rag_bytes_out = make_rag_zip_from_zip_path(
                    repack_local,
                    api_key,
                    rag_chunk_size=task_rag_chunk_size,
                    rag_chunk_overlap=task_rag_chunk_overlap,
                    unit_type=unit_type,
                )
            finally:
                _safe_unlink(repack_local)

            if not rag_bytes_out or len(rag_bytes_out) < _MIN_RAG_ZIP_BYTES:
                item["rag_error"] = "RAG ZIP 產物無效或過小（未產生可用向量庫 ZIP）"
            else:
                save_zip(
                    rag_bytes_out,
                    f"{page_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_page_id=body.rag_page_id,
                    page_id=rag_zip_page_id,
                )
                verify_err = _verify_saved_rag_zip_readable(rag_zip_page_id)
                if verify_err:
                    item["rag_error"] = verify_err
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


def _build_one_rag_zip_output_item(
    body: PackRequest,
    pid: str,
    api_key: str,
    zip_bytes: bytes,
    filename: str,
    *,
    do_rag: bool = True,
    unit_type: int = RAG_UNIT_TYPE_DEFAULT,
    task_rag_chunk_size: int,
    task_rag_chunk_overlap: int,
    transcript_override: str | None = None,
) -> dict[str, Any]:
    """
    將單一 repack 單元上傳至 repack，並依 do_rag 分派建立對應的 rag 產物。
    do_rag 為 True：以 FAISS 建向量庫 RAG ZIP（見 :func:`_build_faiss_rag_item`）。
    do_rag 為 False：repack 照舊，rag 區改上傳逐字稿 transcript.md ZIP 或 repack 同內容複製
    （見 :func:`_build_repack_or_transcript_rag_item`）。
    回傳與 build-rag-zip outputs[] 單筆相同結構；失敗時含 rag_error。
    每筆皆含 **folder_combination**（檔名 stem，寫入 Rag_Unit.folder_combination）、**unit_name**（顯示名，可經請求 unit_names 覆寫）、rag_chunk_size／rag_chunk_overlap（本任務實際使用，供寫入 Rag_Unit）。
    註：bucket 內檔名仍為 stem_rag.zip；請看 output.rag_mode。
    """
    repack_page_id = repack_zip_stem_from_filename(filename) if filename else None
    if not repack_page_id or "\\" in repack_page_id:
        repack_page_id = str(uuid.uuid4())

    if not do_rag:
        return _build_repack_or_transcript_rag_item(
            body,
            pid,
            zip_bytes,
            filename,
            repack_page_id,
            unit_type=unit_type,
            task_rag_chunk_size=task_rag_chunk_size,
            task_rag_chunk_overlap=task_rag_chunk_overlap,
            transcript_override=transcript_override,
        )

    return _build_faiss_rag_item(
        body,
        pid,
        api_key,
        zip_bytes,
        filename,
        repack_page_id,
        unit_type=unit_type,
        task_rag_chunk_size=task_rag_chunk_size,
        task_rag_chunk_overlap=task_rag_chunk_overlap,
    )


def _rag_zip_build_counts(outputs: list[dict[str, Any]]) -> dict[str, int]:
    total = len(outputs)
    failed = sum(1 for o in outputs if o.get("rag_error"))
    return {"total": total, "built_ok": total - failed, "built_failed": failed}


def _rag_unit_row_from_build_output(
    output: dict[str, Any], body: PackRequest, pid: str, course_id: int
) -> dict[str, Any]:
    """由 build-rag-zip 單筆 output 組出 Rag_Unit 列；依 unit_type 帶入 transcript／檔名／youtube_url。"""
    ut = output.get("unit_type")
    try:
        unit_type_val = int(ut) if ut is not None else RAG_UNIT_TYPE_DEFAULT
    except (TypeError, ValueError):
        unit_type_val = RAG_UNIT_TYPE_DEFAULT
    if unit_type_val < 0 or unit_type_val > 4:
        unit_type_val = RAG_UNIT_TYPE_DEFAULT
    unit_transcript = ""
    text_fn = ""
    mp3_fn = ""
    yt_url = ""
    if unit_type_val in (RAG_UNIT_TYPE_TEXT, RAG_UNIT_TYPE_MP3, RAG_UNIT_TYPE_YOUTUBE):
        tp_raw = output.get("transcript_plain")
        tp = tp_raw if isinstance(tp_raw, str) else ("" if tp_raw is None else str(tp_raw))
        if tp.strip():
            unit_transcript = tp
        if unit_type_val == RAG_UNIT_TYPE_TEXT:
            text_fn = output.get("text_file_name") or ""
        if unit_type_val == RAG_UNIT_TYPE_MP3:
            mp3_fn = output.get("mp3_file_name") or ""
            text_fn = output.get("text_file_name") or ""
        if unit_type_val == RAG_UNIT_TYPE_YOUTUBE:
            yt_url = output.get("youtube_url") or ""
            text_fn = output.get("text_file_name") or ""
    try:
        cs_raw = output.get("rag_chunk_size", output.get("chunk_size", body.rag_chunk_size))
        co_raw = output.get("rag_chunk_overlap", output.get("chunk_overlap", body.rag_chunk_overlap))
        cs_out = int(cs_raw)
        co_out = int(co_raw)
    except (TypeError, ValueError):
        cs_out, co_out = _clamp_chunk_pair(body.rag_chunk_size, body.rag_chunk_overlap)
    else:
        cs_out, co_out = _clamp_chunk_pair(cs_out, co_out)
    fc = (output.get("folder_combination") or "").strip() or (output.get("unit_name") or "").strip()
    return _rag_unit_default_row(
        body.rag_page_id,
        pid,
        course_id=course_id,
        unit_name=output.get("unit_name", ""),
        folder_combination=fc,
        unit_type=unit_type_val,
        repack_file_name=output.get("repack_filename", ""),
        rag_file_name=output.get("rag_filename", ""),
        rag_file_size=float(output.get("file_size") or 0),
        transcript=unit_transcript,
        text_file_name=text_fn,
        mp3_file_name=mp3_fn,
        youtube_url=yt_url,
        rag_chunk_size=cs_out,
        rag_chunk_overlap=co_out,
    )


def _persist_rag_build_metadata(body: PackRequest, pid: str, course_id: int, response: dict[str, Any]) -> None:
    """成功建置後更新 Rag 表並為每個成功輸出單元建立 Rag_Unit 記錄。"""
    supabase = get_supabase()
    ts = now_taipei_iso()

    # 先嘗試更新 Rag，即使失敗也繼續寫 Rag_Unit（schema cache 未同步時不中斷整批）
    try:
        update_payload = {
            "rag_metadata": response,
            "updated_at": ts,
        }
        (
            supabase.table("Rag")
            .update(update_payload)
            .eq("rag_page_id", body.rag_page_id)
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .execute()
        )
    except Exception:
        pass

    outputs = response.get("outputs", [])
    for output in outputs:
        if output.get("rag_error"):
            continue
        unit_row = _rag_unit_row_from_build_output(output, body, pid, course_id)
        try:
            insert_rag_child_row("Rag_Unit", unit_row)
        except Exception:
            pass


@router.get("/pages", response_model=ListRagResponse)
def list_rag(
    request: Request,
    person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 Rag.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false",
    ),
):
    """
    列出 Rag 表內容（deleted=False），須傳 course_id，僅回傳該課程的 Rag／Rag_Unit／Rag_Quiz；
    且僅回傳與 query person_id 相符之列，Rag.local 須與 query local 相符（未傳 local 時依連線自動判定）。
    回傳列依 created_at 由舊到新排序。
    每筆 Rag 含 units（Rag_Unit 列表），每個 unit 含 quizzes（Rag_Quiz 列表，含 follow_up、quiz_history_list）。
    音訊單元（unit_type=3）且 mp3_file_name 非空時，另含 mp3_audio_url：相對於 API 根路徑的 GET /rag/page/unit/mp3-file 查詢字串（`rag_page_id`、`rag_unit_id`，不需 person_id），可接在後端 origin 後作為 `<audio src>`。
    YouTube 單元（unit_type=4）且 youtube_url 非空時，另含 youtube_url_api：相對於 API 根路徑的 GET /rag/page/unit/youtube-url 查詢字串（同上，不需 person_id）。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _rag_table_select(
            RAG_SELECT_ALL,
            exclude_deleted=True,
            local_match=local_filter,
            course_id=course_id,
        )
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        rag_page_ids = list(dict.fromkeys(
            r.get("rag_page_id") for r in data if r.get("rag_page_id")
        ))
        units_by_tab = _units_by_rag_page_ids(rag_page_ids, course_id=course_id)

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
        quizzes_by_unit = _quizzes_by_rag_unit_ids(all_unit_ids, course_id=course_id)

        for row in data:
            page_id = row.get("rag_page_id")
            units = units_by_tab.get(page_id, []) if page_id else []
            for unit in units:
                uid = unit.get("rag_unit_id")
                uid_int = int(uid) if uid is not None else None
                unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []
                try:
                    utype = int(unit.get("unit_type") or 0)
                except (TypeError, ValueError):
                    utype = 0
                unit_name_q = (unit.get("unit_name") or "").strip()
                folder_c = (unit.get("folder_combination") or "").strip()
                if (
                    utype == RAG_UNIT_TYPE_MP3
                    and (unit.get("mp3_file_name") or "").strip()
                    and (unit_name_q or folder_c)
                    and page_id
                    and uid_int is not None
                ):
                    unit["mp3_audio_url"] = (
                        "/rag/page/unit/mp3-file?"
                        + urlencode(
                            {
                                "rag_page_id": str(page_id).strip(),
                                "rag_unit_id": str(uid_int),
                                "course_id": str(course_id),
                            }
                        )
                    )
                if (
                    utype == RAG_UNIT_TYPE_YOUTUBE
                    and (unit.get("youtube_url") or "").strip()
                    and page_id
                    and uid_int is not None
                ):
                    unit["youtube_url_api"] = (
                        "/rag/page/unit/youtube-url?"
                        + urlencode(
                            {
                                "rag_page_id": str(page_id).strip(),
                                "rag_unit_id": str(uid_int),
                                "course_id": str(course_id),
                            }
                        )
                    )
            row["units"] = units

        data = to_json_safe(data)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        _logger.exception("GET /rag/pages 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag 失敗: {e!s}")


def _create_rag_record(
    *,
    rag_page_id: str,
    person_id: str,
    tab_name: str,
    course_id: int,
    local: bool = False,
) -> dict[str, Any]:
    """建立一筆 Rag；回傳 create 回應欄位。"""
    supabase = get_supabase()
    r = (
        supabase.table("Rag")
        .insert(
            _rag_default_row(
                rag_page_id,
                tab_name=tab_name,
                person_id=person_id,
                course_id=course_id,
                file_metadata=None,
                local=local,
            )
        )
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=500, detail="新增 Rag 失敗")
    row = r.data[0]
    return {
        "rag_id": row["rag_id"],
        "rag_page_id": row["rag_page_id"],
        "tab_name": row.get("tab_name"),
        "person_id": row.get("person_id"),
        "course_id": row.get("course_id"),
        "local": row.get("local"),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


def _validate_rag_tab_create_fields(
    *,
    rag_page_id: str,
    person_id: str,
    tab_name: str,
    caller_person_id: str,
) -> tuple[str, str, str]:
    """驗證 page/create 與 page/create-upload-zip 共用欄位；回傳 strip 後 (rag_page_id, person_id, tab_name)。"""
    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")
    pid = (person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="person_id 與 query 不一致")
    name = (tab_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    return fid, pid, name


def _upload_rag_zip_contents(
    *,
    contents: bytes,
    filename: str,
    rag_page_id: str,
    person_id: str,
    course_id: int,
) -> dict[str, Any]:
    """上傳 ZIP 並更新 Rag.file_metadata；回傳 file_metadata。"""
    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            folders = get_second_level_folders_from_zip_file(zip_ref)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取 ZIP 檔案")

    supabase = get_supabase()
    r = (
        supabase.table("Rag")
        .select("rag_id, created_at")
        .eq("rag_page_id", rag_page_id)
        .eq("person_id", person_id)
        .eq("course_id", course_id)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_page_id 的 Rag 資料，請先呼叫 POST /rag/page/create 建立",
        )
    row = r.data[0]

    try:
        save_zip(
            contents,
            filename,
            folder=FOLDER_UPLOAD,
            person_id=person_id,
            page_id=rag_page_id,
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
        "rag_page_id": rag_page_id,
        "created_at": to_taipei_iso(row["created_at"]),
        "filename": filename,
        "second_folders": folders,
        "file_size": file_size_mb,
    }
    update_payload: dict[str, Any] = {
        "file_metadata": file_metadata,
        "file_size": file_size_mb,
        "updated_at": now_taipei_iso(),
    }
    try:
        supabase.table("Rag").update(update_payload).eq("rag_page_id", rag_page_id).eq("person_id", person_id).eq("course_id", course_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return file_metadata


@router.post("/page/create")
def create_unit(
    body: openapi_body(
        CreateRagRequest,
        {"rag_page_id": "string", "person_id": "string", "tab_name": "string", "local": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    只建立一筆 Rag 資料，接受 rag_page_id、person_id、tab_name（必填）、local（選填，預設 false）。
    須傳 query course_id，寫入 Rag.course_id。
    回傳新增的 rag_id、rag_page_id、person_id、course_id、tab_name、local、created_at。
    """
    fid, pid, tab_name = _validate_rag_tab_create_fields(
        rag_page_id=body.rag_page_id,
        person_id=body.person_id,
        tab_name=body.tab_name,
        caller_person_id=caller_person_id,
    )
    try:
        return _create_rag_record(
            rag_page_id=fid,
            person_id=pid,
            tab_name=tab_name,
            course_id=course_id,
            local=body.local,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/page/tab-name")
def update_unit_tab_name(
    body: openapi_body(UpdateRagUnitNameRequest, {"rag_id": 1, "tab_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    更新既有 Rag 的 tab_name。以 rag_id（Rag 主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_id、rag_page_id、person_id、tab_name、updated_at。
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
            .select("rag_id, rag_page_id, person_id, course_id")
            .eq("rag_id", body.rag_id)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_id 的 Rag 資料，或已刪除")
        row = sel.data[0]
        fid = row.get("rag_page_id")
        pid = row.get("person_id")
        if ((pid or "").strip() != caller_person_id):
            raise HTTPException(status_code=403, detail="無權修改該 Rag")
        ts = now_taipei_iso()
        supabase.table("Rag").update({"tab_name": tab_name, "updated_at": ts}).eq("rag_id", body.rag_id).eq("deleted", False).execute()
        return {
            "rag_id": body.rag_id,
            "rag_page_id": fid,
            "person_id": pid,
            "tab_name": tab_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _do_delete_rag_file_by_page_id(fid: str, course_id: int) -> tuple[bool, str]:
    """依 rag_page_id、course_id 將 Rag 未刪除列軟刪除，同時軟刪除對應 Rag_Unit，並刪除 storage 資料夾。"""
    supabase = get_supabase()
    sel = (
        supabase.table("Rag")
        .select("person_id")
        .eq("rag_page_id", fid)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .execute()
    )
    if not sel.data:
        raise HTTPException(status_code=404, detail="找不到該 rag_page_id 的 Rag 資料，或已刪除")
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
        (
            supabase.table("Rag")
            .update({"deleted": True, "updated_at": now_taipei_iso()})
            .eq("rag_page_id", fid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Rag 表失敗: {e}")
    try:
        def build_delete_units(with_course_filter: bool):
            q = (
                supabase.table("Rag_Unit")
                .update({"deleted": True, "updated_at": now_taipei_iso()})
                .eq("rag_page_id", fid)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q

        execute_with_course_id_fallback("Rag_Unit", build_delete_units, course_id)
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


@router.put("/page/delete/{rag_page_id}", status_code=200, summary="Delete Rag File", operation_id="rag_tab_delete")
def delete_rag_file(
    _person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="要刪除的 rag_page_id"),
):
    """
    PUT /rag/page/delete/{rag_page_id}。
    軟刪除：將 Rag 表該 rag_page_id 之未刪除列 deleted 設為 true，同時軟刪除所有對應 Rag_Unit，並刪除 storage 資料夾。
    """
    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")
    folder_deleted, pid = _do_delete_rag_file_by_page_id(fid, course_id)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "rag_page_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }


@router.post("/page/create-upload-zip")
async def create_upload_zip(
    caller_person_id: PersonId,
    course_id: CourseId,
    file: UploadFile = File(...),
    rag_page_id: str = Form(..., description="Rag 的 tab 識別，對應 Rag 表 rag_page_id 欄位"),
    person_id: str = Form(..., description="使用者/路徑識別，需與 query person_id 一致"),
    tab_name: str = Form(..., description="Rag 顯示名稱，寫入 Rag 表 tab_name 欄位"),
    local: bool = Form(False, description="是否為本機 RAG，寫入 Rag 表 local 欄位"),
):
    """
    建立 Rag 並上傳 ZIP（先 page/create，再 page/upload-zip）。
    multipart/form-data：file、rag_page_id、person_id、tab_name、local（選填，預設 false）。
    須傳 query course_id、person_id。
    回傳 create 欄位與 file_metadata。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    fid, pid, name = _validate_rag_tab_create_fields(
        rag_page_id=rag_page_id,
        person_id=person_id,
        tab_name=tab_name,
        caller_person_id=caller_person_id,
    )

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    try:
        create_result = _create_rag_record(
            rag_page_id=fid,
            person_id=pid,
            tab_name=name,
            course_id=course_id,
            local=local,
        )
        file_metadata = _upload_rag_zip_contents(
            contents=contents,
            filename=file.filename,
            rag_page_id=fid,
            person_id=pid,
            course_id=course_id,
        )
        return {**create_result, "file_metadata": file_metadata}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/page/upload-zip")
async def upload_zip(
    caller_person_id: PersonId,
    course_id: CourseId,
    file: UploadFile = File(...),
    rag_page_id: str = Form(..., description="對應 page/create 建立的 rag_page_id，ZIP 會存於此路徑"),
    person_id: str = Form(..., description="寫入儲存路徑的 person_id，需與 page/create 一致"),
):
    """
    Upload Zip：只做上傳並寫入資料庫。需先以 page/create 建立該 rag_page_id 的 Rag 資料。
    亦可改用 POST /rag/page/create-upload-zip 一次完成建立與上傳。
    會更新該筆 Rag 的 file_metadata（filename、second_folders、file_size 等）與 file_size 欄位（皆為 MB）。
    回傳 file_metadata。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    resolved_person_id = (person_id or "").strip()
    if not resolved_person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if resolved_person_id != caller_person_id:
        raise HTTPException(status_code=400, detail="Form 的 person_id 與 query 不一致")

    try:
        return _upload_rag_zip_contents(
            contents=contents,
            filename=file.filename,
            rag_page_id=fid,
            person_id=resolved_person_id,
            course_id=course_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/page/build-rag-zip-stream", include_in_schema=False)
@router.post("/page/build-rag-zip")
def build_rag_zip(
    body: openapi_body(
        PackRequest,
        {
            "rag_page_id": "string",
            "person_id": "string",
            "unit_list": "folder1",
            "unit_names": "",
            "unit_types": "",
            "transcripts": None,
            "rag_chunk_size": 1000,
            "rag_chunk_overlap": 200,
            "rag_chunk_sizes": "",
            "rag_chunk_overlaps": "",
            "build_faiss": None,
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    repack_only: bool = Query(
        False,
        description="為 True 時強制不建 FAISS（unit_type=1 時 rag 改為 repack 複製）；不影響 unit_type=2/3/4 之逐字稿 rag ZIP",
    ),
):
    """
    依先前上傳的 ZIP（rag_page_id）與 unit_list 重新打包。
    **FAISS 建置規則（逐 unit 判斷）**：`user_type==1`（且未強制關閉）且該 unit 之 `unit_type==1 (rag)` → 建 FAISS 並上傳至 rag；`unit_type` 為 2／3／4 時仍 repack 原 ZIP，但 **rag 區上傳內含單一 `transcript.md`（逐字稿全文）之 ZIP**，非 repack 複製；其餘 unit_type==0 等 → repack 同內容複製至 rag。
    可選 query **repack_only=true**：強制全部 unit 不建 FAISS；**不影響** 2／3／4 之逐字稿 rag ZIP 行為。
    可選 body **build_faiss**：`false` 同 repack_only；`true` 強制允許 FAISS（仍需 unit_type==1 觸發）；省略時依 user_type 判定。
    LLM API Key 僅在「最終會建 FAISS」（do_rag 為 True）時必填（依 course_id 自 Course_Setting key=rag-api-key 取得；見 PUT /rag/llm_api_key）。
    body.unit_types 為選填，與 unit_list 逗號分段對齊；**未傳或該段為 0** 時會依單元 ZIP 推斷（恰一音訊＋一文字檔→3、僅一個 .md 等→2；**YouTube 仍須明確傳 4**）。寫入各 Rag_Unit.unit_type。**推斷為 2** 且來源為 `.md`/`.txt` 時 **Rag_Unit.transcript** 為檔案 UTF-8 全文（含 Markdown）。
    body.transcripts 為選填，與 unit_list 逗號分段同序；索引 i 之字串若非空白，覆寫該單元逐字稿（Markdown UTF-8 原樣），仍自 ZIP 擷取 text_file_name／mp3_file_name／youtube_url。
    body.unit_names 為選填，與 packed 任務同序（逗號字串或 JSON 字串陣列）；該段非空白時覆寫串流 output.unit_name 與寫入之 Rag_Unit.unit_name（顯示名）。output.folder_combination 恒為 repack ZIP 檔名 stem（寫入 Rag_Unit.folder_combination；多資料夾為 ``a/tb/tc``）。

    **回應為 NDJSON 串流**（`application/x-ndjson`），請以 `fetch` 讀取 `response.body`，勿使用單次 `response.json()`。
    每一輸出單元須 **成功上傳 repack**；rag 資料夾須 **成功寫入**（unit_type=1 且建 FAISS 為向量庫 ZIP；2／3／4 為逐字稿 md ZIP；其餘為 repack 同內容），且**上傳後能自儲存讀回非空檔**。
    整批成功時自動在 Rag_Unit 表建立對應記錄（每個輸出單元一筆）並更新 Rag.rag_metadata。
    整批任一有 `rag_error` 則 `complete.success` 為 false（不寫入 Rag 表，不建立 Rag_Unit）。

    事件列舉（每行一個物件）：
    - `{"type":"start","total":N,"source_rag_page_id":"...","unit_list":"...","user_type":int,"build_faiss_request":bool|null,"repack_only":bool,"allow_faiss":bool}`（allow_faiss=各 unit 是否可建 FAISS，仍需 unit_type==1 才實際建）
    - `{"type":"building","index":i,"total":N,"completed_before":i-1,"filename":"..."}`
    - `{"type":"unit",...,"output":{...}}`：output 含 **folder_combination**（單元 repack ZIP 檔名 stem，寫入 Rag_Unit.folder_combination；多資料夾為 ``folder1/tfolder2``）、**unit_name**（顯示名，可經 unit_names 覆寫）、rag_mode（`faiss`＝向量庫；`transcript_md`＝逐字稿 md ZIP；`repack_copy`＝與 repack 同內容複製）、`transcript_plain`（鍵名沿用舊版；**unit_type=2 且來源為 .md/.txt 時為檔案 UTF-8 全文，Markdown 原樣**，與寫入 Rag_Unit.transcript 一致）；**text_file_name** 僅 **unit_type=2** 有值（來源文字檔檔名）；**mp3_file_name** 僅 3；**youtube_url** 僅 4；**rag_chunk_size**、**rag_chunk_overlap**（本任務實際使用，與 Rag_Unit 一致）；rag_filename（物件鍵仍為 *_rag.zip）
    - `{"type":"complete","success":bool,"total","built_ok","built_failed","source_rag_page_id","unit_list","outputs"}`

    串流階段 HTTP 狀態碼固定 **200**；請以最後一則 `type===complete` 的 `success` 判斷整批成敗。
    `POST /rag/page/build-rag-zip-stream` 與本端點相同，僅自 OpenAPI 隱藏，供舊客戶端相容。
    """
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")

    require_rag_tab_owner(pid, body.rag_page_id, course_id)

    path = _fetch_source_upload_zip_with_retries(pid, body.rag_page_id)

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        _safe_unlink(path)
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.unit_list)
    if not packed:
        _safe_unlink(path)
        raise HTTPException(status_code=400, detail="unit_list 為空或格式錯誤，例：220222+220301")

    api_key = get_rag_api_key(course_id)
    user_type_val = _fetch_user_type(pid, course_id)

    # 允許 FAISS：user_type==1 且未強制關閉；即使允許，unit_type==1 才真正觸發建置
    if repack_only or body.build_faiss is False:
        allow_faiss = False
    elif body.build_faiss is True:
        allow_faiss = True
    else:
        allow_faiss = (user_type_val == 1)

    if allow_faiss and not api_key:
        _safe_unlink(path)
        raise HTTPException(
            status_code=400,
            detail="請設定 RAG API Key：PUT /rag/llm_api_key（Course_Setting key=rag-api-key，依 course_id）",
        )

    total = len(packed)
    unit_types_per_task = _unit_types_per_task(body.unit_types, total)
    chunk_pairs = _chunk_params_per_task(
        body.rag_chunk_sizes,
        body.rag_chunk_overlaps,
        total,
        body.rag_chunk_size,
        body.rag_chunk_overlap,
    )
    unit_name_overrides = _unit_name_overrides_per_task(body.unit_names, total)

    def _do_rag_for_unit(ut: int) -> bool:
        """只有 allow_faiss 且 unit_type==1 (rag) 時才建 FAISS；其餘走 repack 分支（2/3/4 時 rag 為逐字稿 ZIP）。"""
        return allow_faiss and (ut == RAG_UNIT_TYPE_RAG)

    def ndjson_events():
        outputs: list[dict[str, Any]] = []
        try:
            yield _ndjson_line(
                {
                    "type": "start",
                    "total": total,
                    "source_rag_page_id": body.rag_page_id,
                    "unit_list": body.unit_list,
                    "user_type": user_type_val,
                    "build_faiss_request": body.build_faiss,
                    "repack_only": repack_only,
                    "allow_faiss": allow_faiss,
                }
            )
            for idx, (zip_bytes, filename) in enumerate(packed):
                yield _ndjson_line(
                    {
                        "type": "building",
                        "index": idx + 1,
                        "total": total,
                        "completed_before": idx,
                        "filename": filename,
                    }
                )
                ut = unit_types_per_task[idx]
                t_cs, t_co = chunk_pairs[idx]
                ov_list = body.transcripts or []
                transcript_override = ov_list[idx] if idx < len(ov_list) else None
                item = _build_one_rag_zip_output_item(
                    body,
                    pid,
                    api_key or "",
                    zip_bytes,
                    filename,
                    do_rag=_do_rag_for_unit(ut),
                    unit_type=ut,
                    task_rag_chunk_size=t_cs,
                    task_rag_chunk_overlap=t_co,
                    transcript_override=transcript_override,
                )
                name_ov = unit_name_overrides[idx] if idx < len(unit_name_overrides) else None
                if name_ov is not None:
                    item["unit_name"] = name_ov
                try:
                    ut_out = int(item.get("unit_type") or 0)
                except (TypeError, ValueError):
                    ut_out = 0
                if ut_out != RAG_UNIT_TYPE_RAG:
                    item["rag_chunk_size"] = 0
                    item["rag_chunk_overlap"] = 0
                outputs.append(item)
                yield _ndjson_line(
                    {"type": "unit", "index": idx + 1, "total": total, "output": item}
                )
            success = not any(o.get("rag_error") for o in outputs)
            counts = _rag_zip_build_counts(outputs)
            response = {
                "source_rag_page_id": body.rag_page_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if success:
                _persist_rag_build_metadata(body, pid, course_id, response)
            complete_ev: dict[str, Any] = {
                "type": "complete",
                "success": success,
                "source_rag_page_id": body.rag_page_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if not success:
                complete_ev["message"] = "RAG ZIP 建立失敗（請修正後重試）"
            yield _ndjson_line(complete_ev)
        finally:
            _safe_unlink(path)

    return StreamingResponse(
        ndjson_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/page/units")
def list_rag_units(
    _caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = Query(..., description="要列出 Rag_Unit 的 rag_page_id"),
):
    """
    依 rag_page_id 列出所有未刪除的 Rag_Unit，每個 unit 含關聯的 Rag_Quiz（quizzes，含 follow_up）。
    依 created_at 由舊到新排序。
    """
    try:
        fid = (rag_page_id or "").strip()
        if not fid:
            raise HTTPException(status_code=400, detail="請傳入 rag_page_id")
        supabase = get_supabase()

        def build_units_query(with_course_filter: bool):
            q = (
                supabase.table("Rag_Unit")
                .select("*")
                .eq("rag_page_id", fid)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        units_resp = execute_with_course_id_fallback("Rag_Unit", build_units_query, course_id)
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
        quizzes_by_unit = _quizzes_by_rag_unit_ids(unit_ids, course_id=course_id)

        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        units = to_json_safe(units)
        return {"units": units, "count": len(units)}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("GET /rag/units 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag_Unit 失敗: {e!s}")


@router.put("/page/unit/unit-name")
def update_rag_unit_name(
    body: openapi_body(UpdateRagUnitUnitNameRequest, {"rag_unit_id": 1, "unit_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    更新既有 Rag_Unit 的 unit_name。以 rag_unit_id（主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_unit_id、rag_page_id、person_id、unit_name、updated_at。
    """
    if body.rag_unit_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 rag_unit_id")
    unit_name = (body.unit_name or "").strip()
    if not unit_name:
        raise HTTPException(status_code=400, detail="請傳入 unit_name")
    try:
        supabase = get_supabase()

        def build_unit_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Unit",
                "rag_unit_id, rag_page_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", body.rag_unit_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Unit", build_unit_sel, course_id)
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
            "rag_page_id": row.get("rag_page_id"),
            "person_id": pid,
            "unit_name": unit_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/page/unit/mp3-file",
    summary="Rag Tab Unit Mp3 File",
    operation_id="rag_tab_unit_mp3_file",
    response_model=RagUnitMp3FileResponse,
)
def rag_tab_unit_mp3_file(
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab；repack/upload 路徑皆在其下）"),
    rag_unit_id: int = Query(..., gt=0, description="Rag_Unit 主鍵"),
):
    """
    依 rag_page_id 與 rag_unit_id；**僅 Rag_Unit.unit_type=3（音訊單元）** 時回傳原始音訊。
    **不需** query `person_id`；後端依 `rag_page_id` 自 Rag 解析擁有者後讀 Storage。
    **優先**自該單元之 **repack** ZIP（`Rag_Unit.repack_file_name`／Storage `…/repack/{單元}.zip`）內，依 **folder_combination**（無則 **unit_name**）
    路徑段擷取第一個支援的音訊檔（repack 內仍保留上傳時之資料夾名，與 `repack_tasks_to_zips` 一致）。
    repack 無法讀取時**改讀**該 tab 之 **upload** ZIP（與 GET /rag/unit/mp3-file 相同）。
    Storage `…/rag/{tab}_rag.zip` 僅為逐字稿封包，不含原始 mp3。
    """
    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    supabase = get_supabase()

    def fetch_mp3_unit_row(*, include_folder_combination: bool):
        def build(with_course_filter: bool):
            base_cols = (
                "rag_unit_id, rag_page_id, unit_name, folder_combination, unit_type, deleted, repack_file_name, transcript, course_id"
                if include_folder_combination
                else "rag_unit_id, rag_page_id, unit_name, unit_type, deleted, repack_file_name, transcript, course_id"
            )
            cols = select_without_course_id_if_needed("Rag_Unit", base_cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", rag_unit_id)
                .eq("person_id", owner_pid)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        return execute_with_course_id_fallback("Rag_Unit", build, course_id)

    try:
        sel = fetch_mp3_unit_row(include_folder_combination=True)
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            sel = fetch_mp3_unit_row(include_folder_combination=False)
        else:
            raise
    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_unit_id，或與此 rag_page_id／擁有者不一致",
        )
    row = sel.data[0]
    if row.get("deleted"):
        raise HTTPException(status_code=404, detail="該單元已刪除")
    if (row.get("rag_page_id") or "").strip() != tab:
        raise HTTPException(
            status_code=400,
            detail="rag_page_id 與該 rag_unit_id 所屬之 Rag_Unit.rag_page_id 不一致",
        )
    try:
        ut = int(row.get("unit_type") or 0)
    except (TypeError, ValueError):
        ut = 0
    if ut != RAG_UNIT_TYPE_MP3:
        raise HTTPException(
            status_code=400,
            detail=f"僅 unit_type=3（mp3 音訊單元）可使用此端點，目前 unit_type={ut}",
        )
    folder_name = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if not folder_name:
        raise HTTPException(
            status_code=400,
            detail="Rag_Unit.folder_combination 與 unit_name 皆為空，無法對應 repack／upload ZIP 內單元路徑",
        )

    zip_bytes: bytes | None = None
    zip_is_unit_repack = False
    repack_fn = (row.get("repack_file_name") or "").strip()
    repack_err: str | None = None
    if repack_fn:
        try:
            zip_bytes = read_repack_zip_bytes(repack_fn)
            zip_is_unit_repack = True
        except FileNotFoundError as e:
            repack_err = str(e)
        except ValueError as e:
            repack_err = str(e)
        except Exception as e:
            _logger.exception("讀取 repack ZIP 失敗")
            repack_err = str(e)

    if zip_bytes is None:
        try:
            zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            zip_is_unit_repack = False
        except FileNotFoundError as e:
            detail = str(e)
            if repack_err:
                detail = f"{detail}（repack 亦失敗：{repack_err}）"
            raise HTTPException(status_code=404, detail=detail) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _logger.exception("讀取 upload ZIP 失敗")
            raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    from_repack_ok = bool(repack_fn) and repack_err is None
    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip_with_folder_fallback(
            zip_bytes,
            folder_name,
            allow_scan_other_top_folders=zip_is_unit_repack,
        )
    except ValueError as e:
        if from_repack_ok and zip_is_unit_repack:
            try:
                zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            except FileNotFoundError as e2:
                raise HTTPException(
                    status_code=400,
                    detail=f"{e!s}（repack ZIP 內亦無法對應音訊，且無 upload 可備援：{e2!s}）",
                ) from e
            except ValueError as e2:
                raise HTTPException(status_code=400, detail=f"{e!s}（upload 備援：{e2!s}）") from e
            except Exception as e2:
                _logger.exception("讀取 upload ZIP 備援失敗")
                raise HTTPException(
                    status_code=500,
                    detail=f"{e!s}（upload 備援讀取失敗：{e2!s}）",
                ) from e
            try:
                contents, suffix, inner_path = pick_audio_from_upload_zip_with_folder_fallback(
                    zip_bytes,
                    folder_name,
                    allow_scan_other_top_folders=False,
                )
            except ValueError as e3:
                raise HTTPException(status_code=400, detail=str(e3)) from e
        else:
            raise HTTPException(status_code=400, detail=str(e)) from e

    media = audio_media_type_for_suffix(suffix)
    disp_name = Path(inner_path).name
    audio_b64 = base64.b64encode(contents).decode()
    unit_transcript = transcript_from_row(row)
    return RagUnitMp3FileResponse(
        rag_unit_id=rag_unit_id,
        rag_page_id=tab,
        audio_base64=audio_b64,
        media_type=media,
        filename=disp_name,
        transcript=unit_transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/page/unit/youtube-url
# ---------------------------------------------------------------------------


@router.get(
    "/page/unit/youtube-url",
    summary="Rag Tab Unit Youtube Url",
    operation_id="rag_tab_unit_youtube_url",
    response_model=RagUnitYoutubeUrlResponse,
)
def rag_tab_unit_youtube_url(
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab）"),
    rag_unit_id: int = Query(..., gt=0, description="Rag_Unit 主鍵"),
):
    """
    依 rag_page_id 與 rag_unit_id 回傳 **unit_type=4（YouTube 單元）** 之 `youtube_url`。
    **不需** query `person_id`；後端依 `rag_page_id` 自 Rag 解析擁有者。
    youtube_url 為上傳 ZIP 內文字檔所記錄之 YouTube 連結（建 RAG 時由 `extract_transcript_for_rag_build` 擷取並寫入 `Rag_Unit.youtube_url`）。
    """
    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    supabase = get_supabase()

    def build_youtube_sel(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Unit",
            "rag_unit_id, rag_page_id, unit_type, youtube_url, transcript, deleted, course_id",
            with_course_filter,
        )
        q = (
            supabase.table("Rag_Unit")
            .select(cols)
            .eq("rag_unit_id", rag_unit_id)
            .eq("person_id", owner_pid)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    try:
        sel = execute_with_course_id_fallback("Rag_Unit", build_youtube_sel, course_id)
    except Exception as e:
        _logger.exception("GET /rag/page/unit/youtube-url 查詢 Rag_Unit 失敗")
        raise HTTPException(status_code=500, detail=f"查詢失敗: {e!s}") from e

    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_unit_id，或與此 rag_page_id／擁有者不一致",
        )
    row = sel.data[0]
    if row.get("deleted"):
        raise HTTPException(status_code=404, detail="該單元已刪除")
    if (row.get("rag_page_id") or "").strip() != tab:
        raise HTTPException(
            status_code=400,
            detail="rag_page_id 與該 rag_unit_id 所屬之 Rag_Unit.rag_page_id 不一致",
        )
    try:
        ut = int(row.get("unit_type") or 0)
    except (TypeError, ValueError):
        ut = 0
    if ut != RAG_UNIT_TYPE_YOUTUBE:
        raise HTTPException(
            status_code=400,
            detail=f"僅 unit_type=4（YouTube 單元）可使用此端點，目前 unit_type={ut}",
        )
    yt_url = (row.get("youtube_url") or "").strip()
    if not yt_url:
        raise HTTPException(status_code=404, detail="該 YouTube 單元未記錄 youtube_url，請重新建置 RAG")
    unit_transcript = transcript_from_row(row)
    return RagUnitYoutubeUrlResponse(
        rag_unit_id=int(row["rag_unit_id"]),
        rag_page_id=tab,
        youtube_url=yt_url,
        transcript=unit_transcript,
    )


@router.post("/page/unit/quiz/create", summary="Rag Create Quiz (no LLM)", operation_id="rag_create_quiz")
def insert_rag_quiz_row(
    body: openapi_body(InsertRagQuizRowRequest, {"rag_page_id": "string", "rag_unit_id": 1}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    依 `rag_page_id`／`rag_unit_id` 解析 Rag_Unit 後新增一筆空白 `Rag_Quiz`，**不呼叫 LLM**。`rag_quiz_id` 由資料庫自動產生並於回傳中帶出。
    LLM 出題請用 `POST /rag/page/unit/quiz/llm-generate`。
    """
    try:
        supabase = get_supabase()
        req_tab = (body.rag_page_id or "").strip()
        resolved_unit_id = int(body.rag_unit_id or 0)

        u: dict[str, Any] | None = None

        def build_unit_lookup(with_course_filter: bool, *, by_unit_id: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Unit",
                "rag_unit_id, rag_page_id, person_id, unit_name, course_id",
                with_course_filter,
            )
            q = supabase.table("Rag_Unit").select(cols).eq("deleted", False)
            if by_unit_id:
                q = q.eq("rag_unit_id", resolved_unit_id).limit(1)
            else:
                q = q.eq("rag_page_id", req_tab)
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q

        if resolved_unit_id > 0:
            sel = execute_with_course_id_fallback(
                "Rag_Unit",
                lambda with_course: build_unit_lookup(with_course, by_unit_id=True),
                course_id,
            )
            if sel.data:
                u = sel.data[0]
        else:
            if not req_tab:
                raise HTTPException(
                    status_code=400,
                    detail="請傳入 rag_unit_id（>0），或傳入 rag_page_id 且該 tab 下僅有一筆 Rag_Unit",
                )
            sel = execute_with_course_id_fallback(
                "Rag_Unit",
                lambda with_course: build_unit_lookup(with_course, by_unit_id=False),
                course_id,
            )
            rows = sel.data or []
            if len(rows) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"rag_page_id 需唯一對應一筆 Rag_Unit（目前 {len(rows)} 筆），請改傳 rag_unit_id",
                )
            u = rows[0]

        if u is None:
            raise HTTPException(status_code=404, detail="找不到該 rag_unit_id 的 Rag_Unit 資料，或已刪除")

        uid = int(u.get("rag_unit_id") or 0)
        if req_tab and (u.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_unit_id 對應之 Rag_Unit 不一致")
        pid = (u.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權於該 Rag_Unit 新增題目")
        rag_page_id = (u.get("rag_page_id") or "").strip()
        if not rag_page_id:
            raise HTTPException(status_code=400, detail="該 Rag_Unit 的 rag_page_id 為空，無法寫入 Rag_Quiz")
        quiz_name = (u.get("unit_name") or "").strip()
        ts = now_taipei_iso()
        quiz_row: dict[str, Any] = {
            "rag_page_id": rag_page_id,
            "rag_unit_id": uid,
            "person_id": pid,
            "course_id": course_id,
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": "",
            "quiz_content": "",
            "quiz_hint": "",
            "quiz_answer_reference": "",
            "answer_user_prompt_text": "",
            "answer_content": "",
            "answer_critique": None,
            "for_exam": False,
            "follow_up": False,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = insert_rag_child_row("Rag_Quiz", quiz_row)
        if not ins.data or len(ins.data) == 0:
            raise HTTPException(status_code=500, detail="寫入 Rag_Quiz 失敗（無回傳資料）")
        row = ins.data[0]
        ans = (row.get("quiz_answer") or row.get("answer_content") or "") or ""
        return to_json_safe(
            {
                "rag_quiz_id": row.get("rag_quiz_id"),
                "rag_page_id": row.get("rag_page_id"),
                "rag_unit_id": row.get("rag_unit_id"),
                "person_id": row.get("person_id"),
                "quiz_name": row.get("quiz_name"),
                "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
                "quiz_content": row.get("quiz_content"),
                "quiz_hint": row.get("quiz_hint"),
                "quiz_answer_reference": row.get("quiz_answer_reference"),
                "answer_user_prompt_text": row.get("answer_user_prompt_text"),
                "quiz_answer": ans,
                "answer_content": ans,
                "answer_critique": row.get("answer_critique"),
                "for_exam": row.get("for_exam"),
                "follow_up": row.get("follow_up"),
                "deleted": row.get("deleted"),
                "updated_at": row.get("updated_at"),
                "created_at": row.get("created_at"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /rag/page/unit/quiz/create 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/page/unit/quiz/quiz-name", summary="Update Rag Quiz Name", operation_id="rag_tab_unit_quiz_quiz_name")
def update_rag_quiz_name(
    body: openapi_body(UpdateRagQuizQuizNameRequest, {"rag_quiz_id": 1, "quiz_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    更新既有 Rag_Quiz 的 quiz_name。以 rag_quiz_id（主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_quiz_id、rag_page_id、rag_unit_id、person_id、quiz_name、updated_at。
    """
    quiz_name = (body.quiz_name or "").strip()
    if not quiz_name:
        raise HTTPException(status_code=400, detail="請傳入 quiz_name")
    try:
        supabase = get_supabase()

        def build_quiz_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", body.rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz 資料，或已刪除")
        row = sel.data[0]
        pid = (row.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Rag_Quiz")
        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"quiz_name": quiz_name, "updated_at": ts}).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()
        return {
            "rag_quiz_id": body.rag_quiz_id,
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": pid,
            "quiz_name": quiz_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /rag/page/unit/quiz/quiz-name 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/page/unit/quiz/delete/{rag_quiz_id}",
    status_code=200,
    summary="Delete Rag Quiz",
    operation_id="rag_tab_unit_quiz_delete",
)
def delete_rag_quiz(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_quiz_id: int = PathParam(..., gt=0, description="要軟刪除的 Rag_Quiz 主鍵"),
):
    """
    PUT /rag/page/unit/quiz/delete/{rag_quiz_id}。
    軟刪除：將 Rag_Quiz 該列 deleted 設為 true（僅 person_id 與請求者一致且尚未刪除之列）。
    """
    try:
        supabase = get_supabase()

        def build_quiz_delete_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_delete_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz 資料，或已刪除")
        row = sel.data[0]
        pid = (row.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Rag_Quiz")
        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"deleted": True, "updated_at": ts}).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        return {
            "message": "已將 Rag_Quiz 標記為刪除",
            "rag_quiz_id": rag_quiz_id,
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": pid,
            "rag_quiz_updated": True,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /rag/page/unit/quiz/delete/{rag_quiz_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))

