"""
從 System_Setting 表 key=course_name 讀取課程名稱（出題、評分 prompt 用）。
"""

from typing import Optional

from utils.supabase_client import get_supabase

SYSTEM_SETTING_COURSE_NAME_KEY = "course_name"


def get_course_name() -> Optional[str]:
    """
    取得 System_Setting 中 key=course_name 的 value。
    若尚無資料或 value 為空，回傳 None。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", SYSTEM_SETTING_COURSE_NAME_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        name = (resp.data[0].get("value") or "").strip()
        return name if name else None
    except Exception:
        return None


def get_course_name_for_prompt() -> str:
    """供 LLM prompt 使用；未設定時回傳「該課程」避免句子殘缺。"""
    return (get_course_name() or "").strip() or "該課程"
