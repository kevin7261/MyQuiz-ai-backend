"""
Person_Analysis：個人／課程 LLM 分析設定與結果。

person_id：varchar(255) 登入帳號；課程共用指令為空字串（查詢仍相容 legacy bigint user_id）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from postgrest.exceptions import APIError

from utils.db_schema import ACTIVE_DELETED_FILTER, USER_TABLE
from utils.supabase import get_supabase

logger = logging.getLogger(__name__)

PERSON_ANALYSIS_TABLE = "Person_Analysis"
PERSON_ANALYSIS_COLUMNS = (
    "person_analysis_id, person_id, course_id, analysis_prompt_text, analysis_text, "
    "deleted, updated_at, created_at"
)
COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID = ""
LEGACY_COURSE_WIDE_PERSON_ID = 0

PersonIdDbValue = Union[str, int]


def _person_id_for_db(person_id: str | int) -> str:
    return str(person_id).strip()


def _user_id_for_login(login: str) -> Optional[int]:
    if not login or login == COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID:
        return LEGACY_COURSE_WIDE_PERSON_ID
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
    if key == COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID:
        return COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID
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
    if raw == COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID:
        return [COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID, LEGACY_COURSE_WIDE_PERSON_ID]

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
    if uid == LEGACY_COURSE_WIDE_PERSON_ID:
        return COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID
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
    if pid == LEGACY_COURSE_WIDE_PERSON_ID or pid == "0":
        return {**row, "person_id": COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID}
    if isinstance(pid, int) or (isinstance(pid, str) and str(pid).isdigit()):
        login = login_person_id_from_user_id(pid)
        if login:
            return {**row, "person_id": login}
    return row


def _is_person_id_type_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "22p02" in msg or ("bigint" in msg and "person_id" in msg)


def _insert_person_analysis_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """insert；person_id 自動相容 varchar 與 legacy bigint。"""
    base = {
        "course_id": int(row["course_id"]),
        "analysis_prompt_text": row.get("analysis_prompt_text"),
        "analysis_text": row.get("analysis_text"),
        "deleted": row.get("deleted", False),
    }
    login_key = _person_id_for_db(row["person_id"])
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


def _is_instruction_only_row(row: dict[str, Any]) -> bool:
    """教師指令列：analysis_text 為 null 或空字串。"""
    return not (row.get("analysis_text") or "").strip()


def _instruction_text_from_row(row: dict[str, Any]) -> str:
    """僅從指令列取純文字 analysis_prompt_text（不解析 LLM 結果 JSON）。"""
    if not _is_instruction_only_row(row):
        return ""
    return (row.get("analysis_prompt_text") or "").strip()


def fetch_latest_person_analysis_instruction_row(
    person_id: str | int,
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    """最新一筆教師指令列（analysis_text 為 null／空）。"""
    supabase = get_supabase()
    for pid in person_id_db_lookup_keys(person_id):
        try:
            resp = (
                supabase.table(PERSON_ANALYSIS_TABLE)
                .select(PERSON_ANALYSIS_COLUMNS)
                .eq("person_id", pid)
                .eq("course_id", int(course_id))
                .eq("deleted", False)
                .is_("analysis_text", "null")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                row = _normalize_row_person_id(resp.data[0])
                if _is_instruction_only_row(row):
                    return row
        except APIError as e:
            if _is_person_id_type_error(e):
                continue
            logger.exception(
                "fetch_latest_person_analysis_instruction_row failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
        except Exception:
            logger.exception(
                "fetch_latest_person_analysis_instruction_row failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
    return None


def fetch_latest_person_analysis_result_row(
    person_id: str | int,
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    """最新一筆 LLM 分析結果列（analysis_text 非空）。"""
    supabase = get_supabase()
    for pid in person_id_db_lookup_keys(person_id):
        try:
            resp = (
                supabase.table(PERSON_ANALYSIS_TABLE)
                .select(PERSON_ANALYSIS_COLUMNS)
                .eq("person_id", pid)
                .eq("course_id", int(course_id))
                .eq("deleted", False)
                .not_.is_("analysis_text", "null")
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            for row in resp.data or []:
                normalized = _normalize_row_person_id(row)
                if (normalized.get("analysis_text") or "").strip():
                    return normalized
        except APIError as e:
            if _is_person_id_type_error(e):
                continue
            logger.exception(
                "fetch_latest_person_analysis_result_row failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
        except Exception:
            logger.exception(
                "fetch_latest_person_analysis_result_row failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
    return None


def fetch_latest_person_analysis_setting(
    person_id: str | int,
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    supabase = get_supabase()
    for pid in person_id_db_lookup_keys(person_id):
        try:
            resp = (
                supabase.table(PERSON_ANALYSIS_TABLE)
                .select(PERSON_ANALYSIS_COLUMNS)
                .eq("person_id", pid)
                .eq("course_id", int(course_id))
                .eq("deleted", False)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return _normalize_row_person_id(resp.data[0])
        except APIError as e:
            if _is_person_id_type_error(e):
                continue
            logger.exception(
                "fetch_latest_person_analysis_setting failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
        except Exception:
            logger.exception(
                "fetch_latest_person_analysis_setting failed person_id=%s course_id=%s key=%s",
                person_id,
                course_id,
                pid,
            )
            return None
    return None


def fetch_person_analysis_instruction_text(
    person_id: str | int,
    course_id: int | str,
) -> tuple[Optional[int], str]:
    """
    讀取教師分析指令（純文字，僅指令列）。
    回傳 (person_analysis_id, instruction_text)；無紀錄時 (None, "")。
    """
    for pid in (
        _person_id_for_db(person_id),
        COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID,
    ):
        row = fetch_latest_person_analysis_instruction_row(pid, course_id)
        if not row:
            continue
        text = _instruction_text_from_row(row)
        if text:
            raw_id = row.get("person_analysis_id")
            return (int(raw_id) if raw_id is not None else None), text
    return None, ""


def fetch_person_analysis_user_prompt_for_llm(
    login_person_id: str,
    course_id: int | str,
) -> str:
    """LLM 用教師指令：該生指令列優先，其次課程共用；不從 LLM 結果 JSON 反推。"""
    _, student_text = fetch_person_analysis_instruction_text(login_person_id, course_id)
    if student_text:
        return student_text
    _, course_text = fetch_person_analysis_instruction_text(
        COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID, course_id
    )
    return course_text


def save_person_analysis_prompt_instruction(
    person_id: str | int,
    course_id: int | str,
    analysis_prompt_text: str,
) -> Optional[dict[str, Any]]:
    text = (analysis_prompt_text or "").strip()
    if not text:
        return None
    return _insert_person_analysis_row(
        {
            "person_id": _person_id_for_db(person_id),
            "course_id": int(course_id),
            "analysis_prompt_text": text,
            "analysis_text": None,
            "deleted": False,
        }
    )


def save_person_analysis_setting(
    person_id: str | int,
    course_id: int | str,
    analysis_text: str,
) -> Optional[dict[str, Any]]:
    """寫入 LLM 分析結果列；analysis_prompt_text 僅存於教師指令列（API PUT）。"""
    if not (analysis_text or "").strip():
        return None
    return _insert_person_analysis_row(
        {
            "person_id": _person_id_for_db(person_id),
            "course_id": int(course_id),
            "analysis_prompt_text": None,
            "analysis_text": analysis_text,
            "deleted": False,
        }
    )
