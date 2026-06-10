"""
Course_Setting 表：依 course_id + key 讀寫課程級設定（prompt、API key 等）。
"""

from typing import Any, Optional

from utils.supabase import get_supabase
from utils.taipei_time import now_taipei_iso

COURSE_SETTING_TABLE = "Course_Setting"
COURSE_SETTING_COLUMNS = "course_setting_id, course_id, key, value"

COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY = "person_analysis_user_prompt_text"
COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY = "course_analysis_user_prompt_text"
COURSE_SETTING_RAG_API_KEY = "rag-api-key"
COURSE_SETTING_LLM_MODEL = "llm-model"
COURSE_SETTING_EXAM_API_KEY = "exam-api-key"
# Bank（測試題庫）專屬金鑰與模型（與 rag-api-key／llm-model 完全分開）
COURSE_SETTING_BANK_API_KEY = "bank-api-key"
COURSE_SETTING_BANK_LLM_MODEL = "bank-llm-model"
# Quiz（試卷／Test，搭配 bank 出題）專屬金鑰與模型（與 bank-/exam-/rag- 完全分開）
COURSE_SETTING_QUIZ_API_KEY = "quiz-api-key"
COURSE_SETTING_QUIZ_LLM_MODEL = "quiz-llm-model"


def fetch_course_setting_text(key: str, course_id: int) -> str:
    """讀取 Course_Setting value（依 course_id + key）；失敗或無列時回傳空字串。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_SETTING_TABLE)
            .select("value")
            .eq("key", key)
            .eq("course_id", course_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return ""
        return (resp.data[0].get("value") or "").strip()
    except Exception:
        return ""


def fetch_course_setting_value(key: str, course_id: int) -> Optional[str]:
    """讀取 Course_Setting value；無列或空字串時回傳 None。"""
    text = fetch_course_setting_text(key, course_id)
    return text or None


def fetch_course_setting_row(key: str, course_id: int) -> Optional[dict[str, Any]]:
    """讀取 Course_Setting 整列（course_id + key）；無列時回傳 None。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_SETTING_TABLE)
            .select(COURSE_SETTING_COLUMNS)
            .eq("key", key)
            .eq("course_id", course_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return resp.data[0]
    except Exception:
        return None


def upsert_course_setting_and_get_row(
    supabase, key: str, value: str, course_id: int
) -> Optional[dict[str, Any]]:
    """依 course_id + key 新增或更新一筆 Course_Setting，回傳該筆 row（dict）。"""
    now = now_taipei_iso()
    resp = (
        supabase.table(COURSE_SETTING_TABLE)
        .select("course_setting_id")
        .eq("key", key)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if resp.data and len(resp.data) > 0:
        row_id = resp.data[0].get("course_setting_id")
        supabase.table(COURSE_SETTING_TABLE).update({
            "value": value,
            "updated_at": now,
        }).eq("course_setting_id", row_id).execute()
    else:
        supabase.table(COURSE_SETTING_TABLE).insert({
            "course_id": course_id,
            "key": key,
            "value": value,
            "updated_at": now,
            "created_at": now,
        }).execute()
    resp2 = (
        supabase.table(COURSE_SETTING_TABLE)
        .select(COURSE_SETTING_COLUMNS)
        .eq("key", key)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if not resp2.data or len(resp2.data) == 0:
        return None
    return resp2.data[0]
