"""Bank 端點 course_id 驗證與 Bank 擁有者解析（自 utils.rag_course 複製，獨立不與 rag 共用）。"""

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from postgrest.exceptions import APIError

from utils.db_schema import ACTIVE_DELETED_FILTER, USER_COURSE_RELATION_TABLE
from utils.supabase import get_supabase
from utils.taipei_time import now_taipei_iso

# 課程管理者身分：1 管理者、2 教師（皆可設定／編輯／出題／刪除課程內任何題庫）
BANK_MANAGER_USER_TYPES = (1, 2)

# Bank_Unit／Bank_Group／Bank_QA 是否已有 course_id 欄位（None=尚未探測）
_course_id_column_cache: dict[str, bool | None] = {
    "Bank_Unit": None,
    "Bank_Group": None,
    "Bank_QA": None,
}


def is_missing_course_id_column(err: APIError) -> bool:
    return err.code == "42703" and "course_id" in (err.message or "").lower()


def execute_with_course_id_fallback(
    table: str,
    build_query: Callable[[bool], Any],
    course_id: int | None,
) -> Any:
    """執行查詢；若表尚無 course_id 欄位則略過該篩選（bank_page_id 已由 Bank 列限定課程）。"""
    if course_id is None:
        return build_query(False).execute()
    if _course_id_column_cache.get(table) is False:
        return build_query(False).execute()
    try:
        resp = build_query(True).execute()
        _course_id_column_cache[table] = True
        return resp
    except APIError as e:
        if is_missing_course_id_column(e):
            _course_id_column_cache[table] = False
            return build_query(False).execute()
        raise


def omit_course_id_if_unsupported(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """INSERT 用：表尚無 course_id 欄位時自 payload 移除，避免 42703。"""
    _course_id_column_cache[table] = False
    return {k: v for k, v in row.items() if k != "course_id"}


def select_without_course_id_if_needed(table: str, columns: str, with_course: bool) -> str:
    """Bank_Unit 尚無 course_id 欄位時，自 SELECT 移除 course_id。"""
    if with_course and _course_id_column_cache.get(table) is not False:
        return columns
    return ", ".join(
        part.strip()
        for part in columns.split(",")
        if part.strip() and part.strip() != "course_id"
    )


def insert_bank_child_row(table: str, row: dict[str, Any]) -> Any:
    """INSERT Bank_Unit；表尚無 course_id 時自動略過該欄。"""
    supabase = get_supabase()
    try:
        return supabase.table(table).insert(row).execute()
    except APIError as e:
        if is_missing_course_id_column(e):
            return supabase.table(table).insert(omit_course_id_if_unsupported(table, row)).execute()
        raise


def row_course_id(row: dict[str, Any]) -> int:
    try:
        return int(row.get("course_id") or 0)
    except (TypeError, ValueError):
        return 0


def assert_row_course_id(row: dict[str, Any], course_id: int, entity_label: str = "資料") -> None:
    if "course_id" not in row:
        return
    if row_course_id(row) != course_id:
        raise HTTPException(
            status_code=404,
            detail=f"找不到該{entity_label}，或不屬於 course_id={course_id}",
        )


def _user_type_in_course(person_id: str, course_id: int) -> int | None:
    """查 User_Course_Relation.user_type（未刪除）；無列或非有效帳號時回傳 None。"""
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        resp = (
            get_supabase()
            .table(USER_COURSE_RELATION_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .or_(ACTIVE_DELETED_FILTER)
            .order("course_user_id")
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        ut = resp.data[0].get("user_type")
        return int(ut) if ut is not None else None
    except Exception:
        return None


def require_bank_course_manager(caller_person_id: str, course_id: int) -> None:
    """寫入類題庫端點權限：呼叫者須為該課程管理者（user_type 1／2），否則 403。

    題庫權限模型：同課程所有管理者皆可設定／編輯／出題／刪除課程內任何題庫，不再限本人。
    """
    ut = _user_type_in_course(caller_person_id, course_id)
    if ut is None:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    if ut not in BANK_MANAGER_USER_TYPES:
        raise HTTPException(status_code=403, detail="僅課程管理者或教師可設定題庫")


def touch_bank_updater(bank_page_id: str, course_id: int, updater_person_id: str) -> None:
    """將 Bank 該未刪除列的 updater + updated_at 更新為最後修改者。

    供所有 bank-scoped 寫入（含題庫內容：出題／編輯 prompt／批改／刪單元題組 QA）收尾呼叫，
    讓清單反映「最後修改任何欄位的人」。失敗不阻斷主流程。
    """
    rid = (bank_page_id or "").strip()
    pid = (updater_person_id or "").strip()
    if not rid or not pid:
        return
    try:
        (
            get_supabase()
            .table("Bank")
            .update({"updater": pid, "updated_at": now_taipei_iso()})
            .eq("bank_page_id", rid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .execute()
        )
    except Exception:
        pass


def require_bank_tab_owner(
    person_id: str,
    bank_page_id: str,
    course_id: int,
    *,
    require_person_match: bool = True,
) -> dict[str, Any]:
    """確認 Bank 列存在、course_id 一致。回傳 Bank 列。

    權限模型已改為「課程管理者皆可操作任何題庫」，故 person 不再作為擁有者過濾條件；
    require_person_match 保留參數相容，但不再用於 SQL 過濾（呼叫端應另以
    require_bank_course_manager 驗證權限）。
    """
    rid = (bank_page_id or "").strip()
    if not rid or "/" in rid or "\\" in rid:
        raise HTTPException(status_code=400, detail="無效的 bank_page_id")
    supabase = get_supabase()
    sel = (
        supabase.table("Bank")
        .select("bank_page_id, person_id, course_id")
        .eq("bank_page_id", rid)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 bank_page_id，或已刪除／不屬於此 course_id",
        )
    return sel.data[0]


def resolve_bank_tab_owner_person_id(bank_page_id: str, course_id: int) -> str:
    """依 bank_page_id + course_id 取得 upload ZIP 所屬 person_id（不驗證呼叫者 query person_id）。"""
    row = require_bank_tab_owner("", bank_page_id, course_id, require_person_match=False)
    owner = (row.get("person_id") or "").strip()
    if not owner:
        raise HTTPException(status_code=404, detail="找不到該 bank_page_id 之 person_id")
    return owner
