"""routers.zip helpers（自 zip.py 拆分）。"""

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

from fastapi import HTTPException
from postgrest.exceptions import APIError
from storage3.exceptions import StorageApiError


from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
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
    select_without_course_id_if_needed,
)
from utils.rag_transcript import (
    build_transcript_md_zip_bytes,
    extract_transcript_for_rag_build,
    infer_unit_type_when_unspecified,
)


from utils.fs import safe_unlink
from .schemas import PackRequest

_logger = logging.getLogger("routers.zip")


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
        safe_unlink(path)
        return None
    except Exception:
        safe_unlink(path)
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
            safe_unlink(verify_path)
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
                safe_unlink(repack_local)

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
    """驗證 pages/upload-zip 欄位；回傳 strip 後 (rag_page_id, person_id, tab_name)。"""
    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")
    pid = (person_id or "").strip() or (caller_person_id or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="person_id 與呼叫者（token）不一致")
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
            detail="找不到該 rag_page_id 的 Rag 資料，請先呼叫 POST /rag/pages/upload-zip 建立",
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
