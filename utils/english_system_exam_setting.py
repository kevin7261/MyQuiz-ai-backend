"""供測驗用 English System：System_Setting key（english_system_localhost / english_system_deploy）。"""

from fastapi import Request

from utils.rag_exam_setting import is_localhost_request

ENGLISH_SYSTEM_EXAM_SETTING_KEY_LOCALHOST = "english_system_localhost"
ENGLISH_SYSTEM_EXAM_SETTING_KEY_DEPLOY = "english_system_deploy"


def exam_english_system_setting_key(request: Request) -> str:
    """本機連線對應 english_system_localhost，否則 english_system_deploy。"""
    return (
        ENGLISH_SYSTEM_EXAM_SETTING_KEY_LOCALHOST
        if is_localhost_request(request)
        else ENGLISH_SYSTEM_EXAM_SETTING_KEY_DEPLOY
    )


def fetch_exam_english_system_id_from_settings(
    supabase, request: Request
) -> tuple[str, int | None]:
    """
    依連線讀取 System_Setting 中對應 key 的 value，解析為 English_System.system_id（主鍵）。
    回傳 (實際使用的 key, system_id)；無列或無效數字則後者為 None。
    """
    key = exam_english_system_setting_key(request)
    resp = (
        supabase.table("System_Setting")
        .select("value")
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if not resp.data or len(resp.data) == 0:
        return key, None
    raw = (resp.data[0].get("value") or "").strip()
    if not raw:
        return key, None
    try:
        return key, int(raw)
    except ValueError:
        return key, None
