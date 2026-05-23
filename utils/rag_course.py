"""RAG 端點 course_id 驗證與 Rag 擁有者解析。"""

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from postgrest.exceptions import APIError

from utils.supabase_client import get_supabase

# Rag_Unit / Rag_Quiz 是否已有 course_id 欄位（None=尚未探測）
_course_id_column_cache: dict[str, bool | None] = {
    "Rag_Unit": None,
    "Rag_Quiz": None,
}


def is_missing_course_id_column(err: APIError) -> bool:
    return err.code == "42703" and "course_id" in (err.message or "").lower()


def execute_with_course_id_fallback(
    table: str,
    build_query: Callable[[bool], Any],
    course_id: int | None,
) -> Any:
    """執行查詢；若表尚無 course_id 欄位則略過該篩選（rag_tab_id 已由 Rag 列限定課程）。"""
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
    """Rag_Unit／Rag_Quiz 尚無 course_id 欄位時，自 SELECT 移除 course_id。"""
    if with_course and _course_id_column_cache.get(table) is not False:
        return columns
    return ", ".join(
        part.strip()
        for part in columns.split(",")
        if part.strip() and part.strip() != "course_id"
    )


def insert_rag_child_row(table: str, row: dict[str, Any]) -> Any:
    """INSERT Rag_Unit／Rag_Quiz；表尚無 course_id 時自動略過該欄。"""
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


def require_rag_tab_owner(
    person_id: str,
    rag_tab_id: str,
    course_id: int,
    *,
    require_person_match: bool = True,
) -> dict[str, Any]:
    """確認 Rag 列存在、course_id 一致；require_person_match 時 person_id 須一致。回傳 Rag 列。"""
    rid = (rag_tab_id or "").strip()
    if not rid or "/" in rid or "\\" in rid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    supabase = get_supabase()
    q = (
        supabase.table("Rag")
        .select("rag_tab_id, person_id, course_id")
        .eq("rag_tab_id", rid)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
    )
    if require_person_match:
        q = q.eq("person_id", person_id)
    sel = q.execute()
    if not sel.data:
        detail = (
            "找不到該 rag_tab_id，或已刪除／不屬於此 course_id"
            if not require_person_match
            else "找不到該 rag_tab_id，或已刪除／不屬於此 person_id／course_id"
        )
        raise HTTPException(status_code=404, detail=detail)
    return sel.data[0]


def resolve_rag_tab_owner_person_id(rag_tab_id: str, course_id: int) -> str:
    """依 rag_tab_id + course_id 取得 upload ZIP 所屬 person_id（不驗證呼叫者 query person_id）。"""
    row = require_rag_tab_owner("", rag_tab_id, course_id, require_person_match=False)
    owner = (row.get("person_id") or "").strip()
    if not owner:
        raise HTTPException(status_code=404, detail="找不到該 rag_tab_id 之 person_id")
    return owner
