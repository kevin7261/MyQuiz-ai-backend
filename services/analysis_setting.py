"""
登入帳號（person_id）解析共用工具。

供 user-analyses／quiz-analyses 端點將 query 傳入值（person_id 或 user_id）
解析為 User.person_id（登入帳號），並提供查詢／寫入時的 person_id 候選鍵。
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from utils.db_schema import ACTIVE_DELETED_FILTER, USER_TABLE
from utils.supabase import get_supabase

logger = logging.getLogger(__name__)

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
