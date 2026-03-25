"""非 HTTP 層讀取 System_Setting 常用 key（出題、評分 prompt 等）。"""

from typing import Optional

from utils.supabase_client import get_supabase

# 與 routers/system_settings SYSTEM_SETTING_COURSE_NAME_KEY 一致
COURSE_NAME_SETTING_KEY = "course_name"


def get_course_name_setting_value(*, default: str = "本課程") -> str:
    """
    讀取 System_Setting 表 key=course_name 的 value。
    無列、value 為 null 或僅空白時回傳 default。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", COURSE_NAME_SETTING_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return default
        raw: Optional[str] = resp.data[0].get("value")
        if raw is None:
            return default
        s = str(raw).strip()
        return s if s else default
    except Exception:
        return default
