"""
弱點分析設定與結果的資料存取。

對齊 Exam_Quiz 批改（answer_user_prompt_text + answer_critique 同一列 UPDATE）：
- person_id：一律為**呼叫 API 的登入帳號**（必填，不寫空字串）
- analysis_prompt_text：教師分析指令（PUT 寫入）
- analysis_text：弱點報告 Markdown（POST llm-analysis 寫入）

一列／(person_id, course_id)，與 Exam_Quiz 一列一題相同概念。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from postgrest.exceptions import APIError

from utils.db_schema import ACTIVE_DELETED_FILTER, USER_TABLE
from utils.supabase import get_supabase
from utils.taipei_time import now_taipei_iso

logger = logging.getLogger(__name__)

# --- Person_Analysis ---
PERSON_ANALYSIS_TABLE = "Person_Analysis"
PERSON_ANALYSIS_COLUMNS = (
    "person_analysis_id, person_id, course_id, analysis_name, "
    "analysis_prompt_text, analysis_text, deleted, updated_at, created_at"
)
# 僅讀取舊資料 fallback（新寫入不再使用）
LEGACY_COURSE_WIDE_PERSON_ID = ""
LEGACY_COURSE_WIDE_PERSON_ID_INT = 0

PersonIdDbValue = Union[str, int]


def _person_id_for_db(person_id: str | int) -> str:
    return str(person_id).strip()


def _user_id_for_login(login: str) -> Optional[int]:
    if not login or login == LEGACY_COURSE_WIDE_PERSON_ID:
        return LEGACY_COURSE_WIDE_PERSON_ID_INT
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("user_id")
            .eq("person_id", login)
            .or_(ACTIVE_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if resp.data and resp.data[0].get("user_id") is not None:
            return int(resp.data[0]["user_id"])
    except Exception:
        logger.exception("_user_id_for_login failed login=%s", login)
    return None


def resolve_login_person_id(person_id: str | int) -> Optional[str]:
    """將 query 傳入值解析為登入帳號（User.person_id）。"""
    key = _person_id_for_db(person_id)
    if not key:
        return None
    try:
        supabase = get_supabase()
        by_login = (
            supabase.table(USER_TABLE)
            .select("person_id")
            .eq("person_id", key)
            .or_(ACTIVE_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if by_login.data:
            return str(by_login.data[0].get("person_id") or key)
        if key.isdigit():
            by_uid = (
                supabase.table(USER_TABLE)
                .select("person_id")
                .eq("user_id", int(key))
                .or_(ACTIVE_DELETED_FILTER)
                .limit(1)
                .execute()
            )
            if by_uid.data:
                return str(by_uid.data[0].get("person_id") or "").strip() or None
        return key
    except Exception:
        logger.exception("resolve_login_person_id failed person_id=%s", person_id)
        return None


def person_id_db_lookup_keys(person_id: str | int) -> list[PersonIdDbValue]:
    """查詢／寫入時依序嘗試的 person_id 值（varchar 與 legacy bigint）。"""
    raw = _person_id_for_db(person_id)
    if not raw:
        return []
    if raw == LEGACY_COURSE_WIDE_PERSON_ID:
        return [LEGACY_COURSE_WIDE_PERSON_ID, LEGACY_COURSE_WIDE_PERSON_ID_INT]

    login = resolve_login_person_id(raw) or raw
    keys: list[PersonIdDbValue] = [login]
    uid = _user_id_for_login(login)
    if uid is not None and uid not in keys:
        keys.append(uid)
    if raw.isdigit() and int(raw) not in keys:
        keys.append(int(raw))
    return keys


def login_person_id_from_user_id(user_id: int | str) -> Optional[str]:
    if user_id is None:
        return None
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    if uid == LEGACY_COURSE_WIDE_PERSON_ID_INT:
        return LEGACY_COURSE_WIDE_PERSON_ID
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("person_id")
            .eq("user_id", uid)
            .or_(ACTIVE_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if resp.data:
            return str(resp.data[0].get("person_id") or "").strip() or None
    except Exception:
        return None
    return None


def _normalize_row_person_id(row: dict[str, Any]) -> dict[str, Any]:
    pid = row.get("person_id")
    if pid is None:
        return row
    if pid == LEGACY_COURSE_WIDE_PERSON_ID_INT or pid == "0":
        return {**row, "person_id": LEGACY_COURSE_WIDE_PERSON_ID}
    if isinstance(pid, int) or (isinstance(pid, str) and str(pid).isdigit()):
        login = login_person_id_from_user_id(pid)
        if login:
            return {**row, "person_id": login}
    return row


def _is_person_id_type_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "22p02" in msg or ("bigint" in msg and "person_id" in msg)


def _prompt_text_from_row(row: Optional[dict[str, Any]]) -> str:
    if not row:
        return ""
    return (row.get("analysis_prompt_text") or "").strip()


def _fetch_row_by_scope(
    table: str,
    columns: str,
    person_id: str | int,
    course_id: int | str,
    *,
    require_field: Optional[str] = None,
    require_null_field: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    讀取 (person_id, course_id) 最新一筆（updated_at 新到舊）未刪除列。
    require_field：僅取該欄位非 null 的列；require_null_field：僅取該欄位為 null 的列（結果列／規則專用列分流）。
    """
    supabase = get_supabase()
    for pid in person_id_db_lookup_keys(person_id):
        try:
            query = (
                supabase.table(table)
                .select(columns)
                .eq("person_id", pid)
                .eq("course_id", int(course_id))
                .eq("deleted", False)
            )
            if require_field:
                query = query.not_.is_(require_field, "null")
            if require_null_field:
                query = query.is_(require_null_field, "null")
            resp = (
                query
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return _normalize_row_person_id(resp.data[0])
        except APIError as e:
            if _is_person_id_type_error(e):
                continue
            logger.exception(
                "fetch row failed table=%s person_id=%s course_id=%s key=%s",
                table,
                person_id,
                course_id,
                pid,
            )
            return None
        except Exception:
            logger.exception(
                "fetch row failed table=%s person_id=%s course_id=%s key=%s",
                table,
                person_id,
                course_id,
                pid,
            )
            return None
    return None


def _fetch_latest_prompt_row_for_course(
    table: str,
    columns: str,
    course_id: int | str,
    *,
    prefer_person_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """讀取課程分析指令：呼叫者最新 prompt 列 → legacy 空字串列 → 同課最新一筆有 prompt 的列。"""
    if prefer_person_id:
        own = _fetch_row_by_scope(
            table, columns, prefer_person_id, course_id,
            require_field="analysis_prompt_text",
        )
        if _prompt_text_from_row(own):
            return own

    legacy = _fetch_row_by_scope(
        table, columns, LEGACY_COURSE_WIDE_PERSON_ID, course_id,
        require_field="analysis_prompt_text",
    )
    if _prompt_text_from_row(legacy):
        return legacy

    try:
        supabase = get_supabase()
        resp = (
            supabase.table(table)
            .select(columns)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .not_.is_("analysis_prompt_text", "null")
            .order("updated_at", desc=True)
            .limit(10)
            .execute()
        )
        for row in resp.data or []:
            normalized = _normalize_row_person_id(row)
            if _prompt_text_from_row(normalized):
                return normalized
    except Exception:
        logger.exception(
            "_fetch_latest_prompt_row_for_course failed table=%s course_id=%s",
            table,
            course_id,
        )
    return None


def _insert_person_analysis_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    login_key = _person_id_for_db(row["person_id"])
    if not login_key:
        logger.error("Person_Analysis insert rejected: empty caller person_id")
        return None

    base = {
        "course_id": int(row["course_id"]),
        "analysis_name": (row.get("analysis_name") or "").strip(),
        "analysis_prompt_text": row.get("analysis_prompt_text"),
        "analysis_text": row.get("analysis_text"),
        "deleted": row.get("deleted", False),
    }
    errors: list[str] = []
    supabase = get_supabase()

    for pid in person_id_db_lookup_keys(login_key):
        try:
            payload = {**base, "person_id": pid}
            resp = supabase.table(PERSON_ANALYSIS_TABLE).insert(payload).execute()
            if resp.data:
                return _normalize_row_person_id(resp.data[0])
            errors.append(f"person_id={pid!r}: insert returned no data")
        except APIError as e:
            errors.append(f"person_id={pid!r}: {e}")
            if not _is_person_id_type_error(e):
                logger.exception(
                    "Person_Analysis insert failed person_id=%s course_id=%s",
                    pid,
                    row.get("course_id"),
                )
                break
        except Exception as e:
            errors.append(f"person_id={pid!r}: {e}")
            logger.exception(
                "Person_Analysis insert failed person_id=%s course_id=%s",
                pid,
                row.get("course_id"),
            )
            break

    logger.error(
        "Person_Analysis insert exhausted keys login=%s course_id=%s errors=%s",
        login_key,
        row.get("course_id"),
        "; ".join(errors),
    )
    return None


def _update_person_analysis_row(
    person_analysis_id: int,
    patch: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        supabase = get_supabase()
        payload = {**patch, "updated_at": now_taipei_iso()}
        resp = (
            supabase.table(PERSON_ANALYSIS_TABLE)
            .update(payload)
            .eq("person_analysis_id", int(person_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
        logger.error(
            "Person_Analysis update returned no data person_analysis_id=%s",
            person_analysis_id,
        )
    except Exception:
        logger.exception(
            "Person_Analysis update failed person_analysis_id=%s",
            person_analysis_id,
        )
    return None


def fetch_person_analysis_instruction_text(
    caller_person_id: str | int,
    course_id: int | str,
) -> tuple[Optional[int], str]:
    """讀取教師分析指令；優先呼叫者列，其次同課其他列（含 legacy）。"""
    row = _fetch_latest_prompt_row_for_course(
        PERSON_ANALYSIS_TABLE,
        PERSON_ANALYSIS_COLUMNS,
        course_id,
        prefer_person_id=_person_id_for_db(caller_person_id) or None,
    )
    if not row:
        return None, ""
    text = _prompt_text_from_row(row)
    if not text:
        return None, ""
    raw_id = row.get("person_analysis_id")
    return (int(raw_id) if raw_id is not None else None), text


def fetch_person_analysis_user_prompt_for_llm(
    caller_person_id: str | int,
    course_id: int | str,
) -> str:
    """LLM 用教師指令。"""
    _, text = fetch_person_analysis_instruction_text(caller_person_id, course_id)
    return text


def fetch_person_analysis_stored(
    caller_person_id: str | int,
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    """讀取呼叫者最新結果列：prompt（自身或同課 fallback）+ 自身最新 analysis_text 列。"""
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        return None

    row = _fetch_row_by_scope(
        PERSON_ANALYSIS_TABLE,
        PERSON_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_text",
    )
    prompt_row = _fetch_latest_prompt_row_for_course(
        PERSON_ANALYSIS_TABLE,
        PERSON_ANALYSIS_COLUMNS,
        course_id,
        prefer_person_id=caller,
    )
    prompt_text = _prompt_text_from_row(prompt_row)
    analysis_text = (row.get("analysis_text") or "").strip() if row else ""

    if not row and not prompt_text and not analysis_text:
        return None

    primary = row or prompt_row or {}
    return {
        "person_analysis_id": primary.get("person_analysis_id"),
        "person_id": caller,
        "course_id": int(course_id),
        "analysis_name": primary.get("analysis_name"),
        "analysis_prompt_text": prompt_text or None,
        "analysis_text": analysis_text or None,
        "created_at": primary.get("created_at"),
        "updated_at": (row or prompt_row or {}).get("updated_at"),
    }


def fetch_person_analyses_by_person(
    caller_person_id: str | int,
) -> list[dict[str, Any]]:
    """讀取呼叫者所有 Person_Analysis 列（跨課程），updated_at 新到舊。"""
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        return []

    supabase = get_supabase()
    rows: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for pid in person_id_db_lookup_keys(caller):
        try:
            resp = (
                supabase.table(PERSON_ANALYSIS_TABLE)
                .select(PERSON_ANALYSIS_COLUMNS)
                .eq("person_id", pid)
                .eq("deleted", False)
                .order("updated_at", desc=True)
                .execute()
            )
        except APIError as e:
            if _is_person_id_type_error(e):
                continue
            logger.exception(
                "fetch_person_analyses_by_person failed person_id=%s key=%s",
                caller,
                pid,
            )
            return rows
        except Exception:
            logger.exception(
                "fetch_person_analyses_by_person failed person_id=%s key=%s",
                caller,
                pid,
            )
            return rows
        for row in resp.data or []:
            rid = row.get("person_analysis_id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rows.append(_normalize_row_person_id(row))
    rows.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return rows


def soft_delete_person_analysis(person_analysis_id: int) -> Optional[dict[str, Any]]:
    """軟刪除：將 Person_Analysis 該列 deleted 設為 true；找不到未刪除列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(PERSON_ANALYSIS_TABLE)
            .update({"deleted": True, "updated_at": now_taipei_iso()})
            .eq("person_analysis_id", int(person_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
    except Exception:
        logger.exception(
            "soft_delete_person_analysis failed person_analysis_id=%s",
            person_analysis_id,
        )
    return None


def save_person_analysis_prompt_instruction(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_prompt_text: str,
) -> Optional[dict[str, Any]]:
    """PUT 教師指令：更新呼叫者「規則專用列」（prompt 非 null、analysis_text 為 null）；無則新增；不碰結果列快照。"""
    text = (analysis_prompt_text or "").strip()
    if not text:
        return None
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Person_Analysis prompt save rejected: empty caller person_id")
        return None
    existing = _fetch_row_by_scope(
        PERSON_ANALYSIS_TABLE,
        PERSON_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_prompt_text",
        require_null_field="analysis_text",
    )
    row_id = existing.get("person_analysis_id") if existing else None
    if row_id is not None:
        return _update_person_analysis_row(int(row_id), {"analysis_prompt_text": text})
    return _insert_person_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_prompt_text": text,
            "deleted": False,
        }
    )


def add_person_analysis_row(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /person-analysis/add：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Person_Analysis add rejected: empty caller person_id")
        return None
    return _insert_person_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_name": analysis_name,
            "analysis_text": "",
            "deleted": False,
        }
    )


def update_person_analysis_name(
    person_analysis_id: int,
    analysis_name: str,
) -> Optional[dict[str, Any]]:
    """PUT /person-analysis/analysis-name：更新該列 analysis_name；找不到未刪除列時回 None。"""
    return _update_person_analysis_row(
        int(person_analysis_id),
        {"analysis_name": (analysis_name or "").strip()},
    )


def save_person_analysis_setting(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_text: str,
    analysis_prompt_text: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST llm-analysis：寫入呼叫者最新結果列（POST /add 建立之列），連同當次規則快照；無結果列時新增一筆。"""
    if not (analysis_text or "").strip():
        return None
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Person_Analysis result save rejected: empty caller person_id")
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    existing = _fetch_row_by_scope(
        PERSON_ANALYSIS_TABLE,
        PERSON_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_text",
    )
    row_id = existing.get("person_analysis_id") if existing else None
    if row_id is not None:
        return _update_person_analysis_row(
            int(row_id),
            {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
        )
    return _insert_person_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_prompt_text": prompt_snapshot,
            "analysis_text": analysis_text,
            "deleted": False,
        }
    )


# --- Course_Analysis ---

COURSE_ANALYSIS_TABLE = "Course_Analysis"
COURSE_ANALYSIS_COLUMNS = (
    "course_analysis_id, person_id, course_id, analysis_name, "
    "analysis_prompt_text, analysis_text, deleted, updated_at, created_at"
)


def _insert_course_analysis_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    login_key = _person_id_for_db(row["person_id"])
    if not login_key:
        logger.error("Course_Analysis insert rejected: empty caller person_id")
        return None

    base = {
        "course_id": int(row["course_id"]),
        "analysis_name": (row.get("analysis_name") or "").strip(),
        "analysis_prompt_text": row.get("analysis_prompt_text"),
        "analysis_text": row.get("analysis_text"),
        "deleted": row.get("deleted", False),
    }
    errors: list[str] = []
    supabase = get_supabase()

    for pid in person_id_db_lookup_keys(login_key):
        try:
            payload = {**base, "person_id": pid}
            resp = supabase.table(COURSE_ANALYSIS_TABLE).insert(payload).execute()
            if resp.data:
                return _normalize_row_person_id(resp.data[0])
            errors.append(f"person_id={pid!r}: insert returned no data")
        except APIError as e:
            errors.append(f"person_id={pid!r}: {e}")
            if not _is_person_id_type_error(e):
                logger.exception(
                    "Course_Analysis insert failed person_id=%s course_id=%s",
                    pid,
                    row.get("course_id"),
                )
                break
        except Exception as e:
            errors.append(f"person_id={pid!r}: {e}")
            logger.exception(
                "Course_Analysis insert failed person_id=%s course_id=%s",
                pid,
                row.get("course_id"),
            )
            break

    logger.error(
        "Course_Analysis insert exhausted keys login=%s course_id=%s errors=%s",
        login_key,
        row.get("course_id"),
        "; ".join(errors),
    )
    return None


def _update_course_analysis_row(
    course_analysis_id: int,
    patch: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        supabase = get_supabase()
        payload = {**patch, "updated_at": now_taipei_iso()}
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .update(payload)
            .eq("course_analysis_id", int(course_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
        logger.error(
            "Course_Analysis update returned no data course_analysis_id=%s",
            course_analysis_id,
        )
    except Exception:
        logger.exception(
            "Course_Analysis update failed course_analysis_id=%s",
            course_analysis_id,
        )
    return None


def fetch_course_analysis_instruction_text(
    caller_person_id: str | int,
    course_id: int | str,
) -> tuple[Optional[int], str]:
    row = _fetch_latest_prompt_row_for_course(
        COURSE_ANALYSIS_TABLE,
        COURSE_ANALYSIS_COLUMNS,
        course_id,
        prefer_person_id=_person_id_for_db(caller_person_id) or None,
    )
    if not row:
        return None, ""
    text = _prompt_text_from_row(row)
    if not text:
        return None, ""
    raw_id = row.get("course_analysis_id")
    return (int(raw_id) if raw_id is not None else None), text


def fetch_course_analysis_user_prompt_for_llm(
    caller_person_id: str | int,
    course_id: int | str,
) -> str:
    _, text = fetch_course_analysis_instruction_text(caller_person_id, course_id)
    return text


def fetch_course_analysis_stored(
    caller_person_id: str | int,
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        return None

    row = _fetch_row_by_scope(
        COURSE_ANALYSIS_TABLE,
        COURSE_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_text",
    )
    prompt_row = _fetch_latest_prompt_row_for_course(
        COURSE_ANALYSIS_TABLE,
        COURSE_ANALYSIS_COLUMNS,
        course_id,
        prefer_person_id=caller,
    )
    prompt_text = _prompt_text_from_row(prompt_row)
    analysis_text = (row.get("analysis_text") or "").strip() if row else ""

    if not row and not prompt_text and not analysis_text:
        return None

    primary = row or prompt_row or {}
    return {
        "course_analysis_id": primary.get("course_analysis_id"),
        "person_id": caller,
        "course_id": int(course_id),
        "analysis_name": primary.get("analysis_name"),
        "analysis_prompt_text": prompt_text or None,
        "analysis_text": analysis_text or None,
        "created_at": primary.get("created_at"),
        "updated_at": (row or prompt_row or {}).get("updated_at"),
    }


def fetch_course_analyses_by_course(
    course_id: int | str,
) -> list[dict[str, Any]]:
    """讀取課程所有 Course_Analysis 列（跨使用者），updated_at 新到舊。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .select(COURSE_ANALYSIS_COLUMNS)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .order("updated_at", desc=True)
            .execute()
        )
        return [_normalize_row_person_id(row) for row in resp.data or []]
    except Exception:
        logger.exception(
            "fetch_course_analyses_by_course failed course_id=%s", course_id
        )
        return []


def soft_delete_course_analysis(course_analysis_id: int) -> Optional[dict[str, Any]]:
    """軟刪除：將 Course_Analysis 該列 deleted 設為 true；找不到未刪除列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .update({"deleted": True, "updated_at": now_taipei_iso()})
            .eq("course_analysis_id", int(course_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
    except Exception:
        logger.exception(
            "soft_delete_course_analysis failed course_analysis_id=%s",
            course_analysis_id,
        )
    return None


def save_course_analysis_prompt_instruction(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_prompt_text: str,
) -> Optional[dict[str, Any]]:
    """PUT 教師指令：更新呼叫者「規則專用列」（prompt 非 null、analysis_text 為 null）；無則新增；不碰結果列快照。"""
    text = (analysis_prompt_text or "").strip()
    if not text:
        return None
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Course_Analysis prompt save rejected: empty caller person_id")
        return None
    existing = _fetch_row_by_scope(
        COURSE_ANALYSIS_TABLE,
        COURSE_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_prompt_text",
        require_null_field="analysis_text",
    )
    row_id = existing.get("course_analysis_id") if existing else None
    if row_id is not None:
        return _update_course_analysis_row(int(row_id), {"analysis_prompt_text": text})
    return _insert_course_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_prompt_text": text,
            "deleted": False,
        }
    )


def add_course_analysis_row(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /course-analysis/add：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Course_Analysis add rejected: empty caller person_id")
        return None
    return _insert_course_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_name": analysis_name,
            "analysis_text": "",
            "deleted": False,
        }
    )


def update_course_analysis_name(
    course_analysis_id: int,
    analysis_name: str,
) -> Optional[dict[str, Any]]:
    """PUT /course-analysis/analysis-name：更新該列 analysis_name；找不到未刪除列時回 None。"""
    return _update_course_analysis_row(
        int(course_analysis_id),
        {"analysis_name": (analysis_name or "").strip()},
    )


def save_course_analysis_setting(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_text: str,
    analysis_prompt_text: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST llm-analysis：寫入呼叫者最新結果列（POST /add 建立之列），連同當次規則快照；無結果列時新增一筆。"""
    if not (analysis_text or "").strip():
        return None
    caller = _person_id_for_db(caller_person_id)
    if not caller:
        logger.error("Course_Analysis result save rejected: empty caller person_id")
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    existing = _fetch_row_by_scope(
        COURSE_ANALYSIS_TABLE,
        COURSE_ANALYSIS_COLUMNS,
        caller,
        course_id,
        require_field="analysis_text",
    )
    row_id = existing.get("course_analysis_id") if existing else None
    if row_id is not None:
        return _update_course_analysis_row(
            int(row_id),
            {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
        )
    return _insert_course_analysis_row(
        {
            "person_id": caller,
            "course_id": int(course_id),
            "analysis_prompt_text": prompt_snapshot,
            "analysis_text": analysis_text,
            "deleted": False,
        }
    )


# 相容舊 import
COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID = LEGACY_COURSE_WIDE_PERSON_ID
