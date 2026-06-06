"""
弱點分析結果列的資料存取。

對齊 Exam／Rag「一列一 page」模式：
- 一列＝一筆分析紀錄（結果列）；所有操作（改名、刪除、寫入報告）一律按主鍵 UPDATE
- person_id：一律為**呼叫 API 的登入帳號**（必填，不寫空字串）
- analysis_prompt_text：產生報告當下的規則快照（POST llm-analysis 寫入）
- analysis_text：弱點報告 Markdown（POST llm-analysis 寫入；POST /add 先建 '' 空白列）

分析規則（教師指令）存於 Course_Setting（key=person_analysis_user_prompt_text／
course_analysis_user_prompt_text，依 course_id），不再佔用分析表的列。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from postgrest.exceptions import APIError

from utils.course_setting import (
    COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY,
    COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY,
    fetch_course_setting_text,
)
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


def fetch_person_analysis_user_prompt_for_llm(course_id: int | str) -> str:
    """LLM 用教師指令（Course_Setting key=person_analysis_user_prompt_text）。"""
    return fetch_course_setting_text(
        COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY, int(course_id)
    )


def fetch_person_analyses_by_person(
    caller_person_id: str | int,
) -> list[dict[str, Any]]:
    """讀取呼叫者所有 Person_Analysis 結果列（跨課程；analysis_text 非 null），依 person_analysis_id 升冪。"""
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
                .not_.is_("analysis_text", "null")
                .order("person_analysis_id", desc=False)
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
    rows.sort(key=lambda r: int(r.get("person_analysis_id") or 0))
    return rows


def fetch_person_analysis_row(person_analysis_id: int) -> Optional[dict[str, Any]]:
    """按主鍵讀取未刪除的 Person_Analysis 列；無列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(PERSON_ANALYSIS_TABLE)
            .select(PERSON_ANALYSIS_COLUMNS)
            .eq("person_analysis_id", int(person_analysis_id))
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
    except Exception:
        logger.exception(
            "fetch_person_analysis_row failed person_analysis_id=%s",
            person_analysis_id,
        )
    return None


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


def add_person_analysis_row(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /person-analyses：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
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
    """PATCH /person-analyses/{person_analysis_id}：更新該列 analysis_name；找不到未刪除列時回 None。"""
    return _update_person_analysis_row(
        int(person_analysis_id),
        {"analysis_name": (analysis_name or "").strip()},
    )


def save_person_analysis_result(
    person_analysis_id: int,
    analysis_text: str,
    analysis_prompt_text: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /{id}/llm-analysis：按主鍵將報告與當次規則快照寫入指定結果列；找不到未刪除列時回 None。"""
    if not (analysis_text or "").strip():
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    return _update_person_analysis_row(
        int(person_analysis_id),
        {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
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


def fetch_course_analysis_user_prompt_for_llm(course_id: int | str) -> str:
    """LLM 用教師指令（Course_Setting key=course_analysis_user_prompt_text）。"""
    return fetch_course_setting_text(
        COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY, int(course_id)
    )


def fetch_course_analyses_by_course(
    course_id: int | str,
) -> list[dict[str, Any]]:
    """讀取課程所有 Course_Analysis 結果列（跨使用者；analysis_text 非 null），依 course_analysis_id 升冪。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .select(COURSE_ANALYSIS_COLUMNS)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .not_.is_("analysis_text", "null")
            .order("course_analysis_id", desc=False)
            .execute()
        )
        return [_normalize_row_person_id(row) for row in resp.data or []]
    except Exception:
        logger.exception(
            "fetch_course_analyses_by_course failed course_id=%s", course_id
        )
        return []


def fetch_course_analysis_row(course_analysis_id: int) -> Optional[dict[str, Any]]:
    """按主鍵讀取未刪除的 Course_Analysis 列；無列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .select(COURSE_ANALYSIS_COLUMNS)
            .eq("course_analysis_id", int(course_analysis_id))
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if resp.data:
            return _normalize_row_person_id(resp.data[0])
    except Exception:
        logger.exception(
            "fetch_course_analysis_row failed course_analysis_id=%s",
            course_analysis_id,
        )
    return None


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


def add_course_analysis_row(
    caller_person_id: str | int,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /course-analyses：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
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
    """PATCH /course-analyses/{course_analysis_id}：更新該列 analysis_name；找不到未刪除列時回 None。"""
    return _update_course_analysis_row(
        int(course_analysis_id),
        {"analysis_name": (analysis_name or "").strip()},
    )


def save_course_analysis_result(
    course_analysis_id: int,
    analysis_text: str,
    analysis_prompt_text: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /{id}/llm-analysis：按主鍵將報告與當次規則快照寫入指定結果列；找不到未刪除列時回 None。"""
    if not (analysis_text or "").strip():
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    return _update_course_analysis_row(
        int(course_analysis_id),
        {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
    )
