"""
Quiz 模組弱點分析結果列資料存取（User_Analysis、Quiz_Analysis）。

User_Analysis：個人弱點分析（基於 Quiz_QA，依 person_id + course_id 範圍）。
  - 定位等同 Person_Analysis 之於 Exam_Quiz。
Quiz_Analysis：測驗課程分析（基於 Quiz_QA，依 course_id 範圍，全體學生）。
  - 定位等同 Course_Analysis 之於 Exam_Quiz。

User_Analysis：分析指令存 Course_Setting（GET/PUT /user-analyses/analysis-user-prompt-text）；
  analysis_prompt_text 為產生報告當下的規則快照。
Quiz_Analysis：分析指令存 Course_Setting（GET/PUT /quiz-analyses/analysis-user-prompt-text）；
  analysis_prompt_text 為產生報告當下的規則快照。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from utils.course_setting import (
    COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT,
    COURSE_SETTING_USER_ANALYSIS_USER_PROMPT_TEXT,
    fetch_course_setting_text,
)
from utils.supabase import get_supabase
from utils.taipei_time import now_taipei_iso

logger = logging.getLogger(__name__)

# ============================================================
# User_Analysis
# ============================================================

USER_ANALYSIS_TABLE = "User_Analysis"
USER_ANALYSIS_COLUMNS = (
    "user_analysis_id, person_id, course_id, analysis_name, "
    "analysis_prompt_text, analysis_text, deleted, updated_at, created_at"
)


def _str(value: Any) -> str:
    return str(value or "").strip()


def _update_user_analysis_row(
    user_analysis_id: int,
    patch: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        supabase = get_supabase()
        payload = {**patch, "updated_at": now_taipei_iso()}
        resp = (
            supabase.table(USER_ANALYSIS_TABLE)
            .update(payload)
            .eq("user_analysis_id", int(user_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return resp.data[0]
        logger.error(
            "User_Analysis update returned no data user_analysis_id=%s",
            user_analysis_id,
        )
    except Exception:
        logger.exception(
            "User_Analysis update failed user_analysis_id=%s", user_analysis_id
        )
    return None


def fetch_user_analyses_by_person(
    caller_person_id: str,
    course_id: int | str,
) -> list[dict[str, Any]]:
    """
    讀取呼叫者在特定課程的所有 User_Analysis 結果列（analysis_text 非 null，deleted=false），
    依 user_analysis_id 升冪。
    """
    pid = _str(caller_person_id)
    if not pid:
        return []
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_ANALYSIS_TABLE)
            .select(USER_ANALYSIS_COLUMNS)
            .eq("person_id", pid)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .not_.is_("analysis_text", "null")
            .order("user_analysis_id", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        logger.exception(
            "fetch_user_analyses_by_person failed person_id=%s course_id=%s",
            pid,
            course_id,
        )
        return []


def fetch_user_analysis_row(user_analysis_id: int) -> Optional[dict[str, Any]]:
    """按主鍵讀取未刪除 User_Analysis 列；無列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_ANALYSIS_TABLE)
            .select(USER_ANALYSIS_COLUMNS)
            .eq("user_analysis_id", int(user_analysis_id))
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception:
        logger.exception(
            "fetch_user_analysis_row failed user_analysis_id=%s", user_analysis_id
        )
    return None


def add_user_analysis_row(
    caller_person_id: str,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /user-analyses：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
    pid = _str(caller_person_id)
    if not pid:
        logger.error("User_Analysis add rejected: empty caller person_id")
        return None
    try:
        supabase = get_supabase()
        payload = {
            "person_id": pid,
            "course_id": int(course_id),
            "analysis_name": _str(analysis_name),
            "analysis_text": "",
            "deleted": False,
        }
        resp = supabase.table(USER_ANALYSIS_TABLE).insert(payload).execute()
        if resp.data:
            return resp.data[0]
        logger.error("User_Analysis insert returned no data person_id=%s", pid)
    except Exception:
        logger.exception(
            "User_Analysis add failed person_id=%s course_id=%s", pid, course_id
        )
    return None


def update_user_analysis_name(
    user_analysis_id: int,
    analysis_name: str,
) -> Optional[dict[str, Any]]:
    """PATCH /user-analyses/{user_analysis_id}：更新該列 analysis_name。"""
    return _update_user_analysis_row(
        int(user_analysis_id),
        {"analysis_name": _str(analysis_name)},
    )


def soft_delete_user_analysis(user_analysis_id: int) -> Optional[dict[str, Any]]:
    """軟刪除：將 User_Analysis 該列 deleted 設為 true。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_ANALYSIS_TABLE)
            .update({"deleted": True, "updated_at": now_taipei_iso()})
            .eq("user_analysis_id", int(user_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception:
        logger.exception(
            "soft_delete_user_analysis failed user_analysis_id=%s", user_analysis_id
        )
    return None


def fetch_user_analysis_user_prompt_for_llm(course_id: int | str) -> str:
    """LLM 用教師指令（Course_Setting key=user_analysis_user_prompt_text）。"""
    return fetch_course_setting_text(
        COURSE_SETTING_USER_ANALYSIS_USER_PROMPT_TEXT, int(course_id)
    )


def fetch_quiz_analysis_user_prompt_for_llm(course_id: int | str) -> str:
    """LLM 用教師指令（Course_Setting key=quiz_analysis_user_prompt_text）。"""
    return fetch_course_setting_text(
        COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT, int(course_id)
    )


def save_user_analysis_result(
    user_analysis_id: int,
    analysis_text: str,
    analysis_prompt_text: Optional[str],
) -> Optional[dict[str, Any]]:
    """POST /{id}/llm-analysis：按主鍵將報告與當次規則快照寫入指定結果列。"""
    if not (analysis_text or "").strip():
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    return _update_user_analysis_row(
        int(user_analysis_id),
        {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
    )


# ============================================================
# Quiz_Analysis
# ============================================================

QUIZ_ANALYSIS_TABLE = "Quiz_Analysis"
QUIZ_ANALYSIS_COLUMNS = (
    "quiz_analysis_id, person_id, course_id, analysis_name, "
    "analysis_prompt_text, analysis_text, deleted, updated_at, created_at"
)


def _update_quiz_analysis_row(
    quiz_analysis_id: int,
    patch: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        supabase = get_supabase()
        payload = {**patch, "updated_at": now_taipei_iso()}
        resp = (
            supabase.table(QUIZ_ANALYSIS_TABLE)
            .update(payload)
            .eq("quiz_analysis_id", int(quiz_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return resp.data[0]
        logger.error(
            "Quiz_Analysis update returned no data quiz_analysis_id=%s",
            quiz_analysis_id,
        )
    except Exception:
        logger.exception(
            "Quiz_Analysis update failed quiz_analysis_id=%s", quiz_analysis_id
        )
    return None


def fetch_quiz_analyses_by_course(
    course_id: int | str,
) -> list[dict[str, Any]]:
    """
    讀取課程所有 Quiz_Analysis 結果列（analysis_text 非 null，deleted=false），
    依 quiz_analysis_id 升冪。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_ANALYSIS_TABLE)
            .select(QUIZ_ANALYSIS_COLUMNS)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .not_.is_("analysis_text", "null")
            .order("quiz_analysis_id", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        logger.exception(
            "fetch_quiz_analyses_by_course failed course_id=%s", course_id
        )
        return []


def fetch_quiz_analysis_row(quiz_analysis_id: int) -> Optional[dict[str, Any]]:
    """按主鍵讀取未刪除 Quiz_Analysis 列；無列時回 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_ANALYSIS_TABLE)
            .select(QUIZ_ANALYSIS_COLUMNS)
            .eq("quiz_analysis_id", int(quiz_analysis_id))
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception:
        logger.exception(
            "fetch_quiz_analysis_row failed quiz_analysis_id=%s", quiz_analysis_id
        )
    return None


def add_quiz_analysis_row(
    caller_person_id: str,
    course_id: int | str,
    analysis_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """POST /quiz-analyses：新增一筆空白結果列（analysis_text=''），供 llm-analysis 填入。"""
    pid = _str(caller_person_id)
    if not pid:
        logger.error("Quiz_Analysis add rejected: empty caller person_id")
        return None
    try:
        supabase = get_supabase()
        payload = {
            "person_id": pid,
            "course_id": int(course_id),
            "analysis_name": _str(analysis_name),
            "analysis_text": "",
            "deleted": False,
        }
        resp = supabase.table(QUIZ_ANALYSIS_TABLE).insert(payload).execute()
        if resp.data:
            return resp.data[0]
        logger.error("Quiz_Analysis insert returned no data person_id=%s", pid)
    except Exception:
        logger.exception(
            "Quiz_Analysis add failed person_id=%s course_id=%s", pid, course_id
        )
    return None


def update_quiz_analysis_name(
    quiz_analysis_id: int,
    analysis_name: str,
) -> Optional[dict[str, Any]]:
    """PATCH /quiz-analyses/{quiz_analysis_id}：更新該列 analysis_name。"""
    return _update_quiz_analysis_row(
        int(quiz_analysis_id),
        {"analysis_name": _str(analysis_name)},
    )


def soft_delete_quiz_analysis(quiz_analysis_id: int) -> Optional[dict[str, Any]]:
    """軟刪除：將 Quiz_Analysis 該列 deleted 設為 true。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_ANALYSIS_TABLE)
            .update({"deleted": True, "updated_at": now_taipei_iso()})
            .eq("quiz_analysis_id", int(quiz_analysis_id))
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception:
        logger.exception(
            "soft_delete_quiz_analysis failed quiz_analysis_id=%s", quiz_analysis_id
        )
    return None


def save_quiz_analysis_result(
    quiz_analysis_id: int,
    analysis_text: str,
    analysis_prompt_text: Optional[str],
) -> Optional[dict[str, Any]]:
    """POST /{id}/llm-analysis：按主鍵將報告與當次規則快照寫入指定結果列。"""
    if not (analysis_text or "").strip():
        return None
    prompt_snapshot = (analysis_prompt_text or "").strip() or None
    return _update_quiz_analysis_row(
        int(quiz_analysis_id),
        {"analysis_text": analysis_text, "analysis_prompt_text": prompt_snapshot},
    )
