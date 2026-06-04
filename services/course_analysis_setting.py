"""
Course_Analysis：課程 LLM 分析設定與結果（person_id 空字串，僅以 course_id 區分）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from utils.supabase import get_supabase

logger = logging.getLogger(__name__)

COURSE_ANALYSIS_TABLE = "Course_Analysis"
COURSE_ANALYSIS_COLUMNS = (
    "course_analysis_id, person_id, course_id, analysis_prompt_text, analysis_text, "
    "deleted, updated_at, created_at"
)
COURSE_WIDE_COURSE_ANALYSIS_PERSON_ID = ""


def _is_instruction_only_row(row: dict[str, Any]) -> bool:
    """教師指令列：analysis_text 為 null 或空字串。"""
    return not (row.get("analysis_text") or "").strip()


def _instruction_text_from_row(row: dict[str, Any]) -> str:
    if not _is_instruction_only_row(row):
        return ""
    return (row.get("analysis_prompt_text") or "").strip()


def _insert_course_analysis_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    base = {
        "person_id": COURSE_WIDE_COURSE_ANALYSIS_PERSON_ID,
        "course_id": int(row["course_id"]),
        "analysis_prompt_text": row.get("analysis_prompt_text"),
        "analysis_text": row.get("analysis_text"),
        "deleted": row.get("deleted", False),
    }
    try:
        supabase = get_supabase()
        resp = supabase.table(COURSE_ANALYSIS_TABLE).insert(base).execute()
        if resp.data:
            return resp.data[0]
        logger.error(
            "Course_Analysis insert returned no data course_id=%s",
            row.get("course_id"),
        )
    except Exception:
        logger.exception(
            "Course_Analysis insert failed course_id=%s",
            row.get("course_id"),
        )
    return None


def fetch_latest_course_analysis_instruction_row(
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    """最新一筆教師指令列（analysis_text 為 null／空）。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .select(COURSE_ANALYSIS_COLUMNS)
            .eq("person_id", COURSE_WIDE_COURSE_ANALYSIS_PERSON_ID)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .is_("analysis_text", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            if _is_instruction_only_row(row):
                return row
    except Exception:
        logger.exception(
            "fetch_latest_course_analysis_instruction_row failed course_id=%s",
            course_id,
        )
    return None


def fetch_latest_course_analysis_result_row(
    course_id: int | str,
) -> Optional[dict[str, Any]]:
    """最新一筆 LLM 分析結果列（analysis_text 非空）。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_ANALYSIS_TABLE)
            .select(COURSE_ANALYSIS_COLUMNS)
            .eq("person_id", COURSE_WIDE_COURSE_ANALYSIS_PERSON_ID)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .not_.is_("analysis_text", "null")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        for row in resp.data or []:
            if (row.get("analysis_text") or "").strip():
                return row
    except Exception:
        logger.exception(
            "fetch_latest_course_analysis_result_row failed course_id=%s",
            course_id,
        )
    return None


def fetch_course_analysis_instruction_text(
    course_id: int | str,
) -> tuple[Optional[int], str]:
    """讀取教師分析指令（純文字，僅指令列）。回傳 (course_analysis_id, instruction_text)。"""
    row = fetch_latest_course_analysis_instruction_row(course_id)
    if not row:
        return None, ""
    text = _instruction_text_from_row(row)
    if not text:
        return None, ""
    raw_id = row.get("course_analysis_id")
    return (int(raw_id) if raw_id is not None else None), text


def fetch_course_analysis_user_prompt_for_llm(course_id: int | str) -> str:
    """LLM 用教師指令（僅指令列，不從 LLM 結果 JSON 反推）。"""
    _, text = fetch_course_analysis_instruction_text(course_id)
    return text


def save_course_analysis_prompt_instruction(
    course_id: int | str,
    analysis_prompt_text: str,
) -> Optional[dict[str, Any]]:
    text = (analysis_prompt_text or "").strip()
    if not text:
        return None
    return _insert_course_analysis_row(
        {
            "course_id": int(course_id),
            "analysis_prompt_text": text,
            "analysis_text": None,
            "deleted": False,
        }
    )


def save_course_analysis_setting(
    course_id: int | str,
    analysis_text: str,
) -> Optional[dict[str, Any]]:
    """寫入 LLM 分析結果列；analysis_prompt_text 僅存於教師指令列（API PUT）。"""
    if not (analysis_text or "").strip():
        return None
    return _insert_course_analysis_row(
        {
            "course_id": int(course_id),
            "analysis_prompt_text": None,
            "analysis_text": analysis_text,
            "deleted": False,
        }
    )
