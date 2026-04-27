"""
LLM / Deepgram 等 API Key 工具模組。
從 User 表依 person_id 取得個人 LLM API Key；或從 System_Setting 取得系統設定（llm_api_key、deepgram_api_key 等）。
"""

# 引入 Optional 型別，表示可為 None
from typing import Optional

# 引入 Supabase 客戶端取得函數
from utils.db_tables import USER_TABLE
from utils.supabase_client import get_supabase

# System_Setting 表：key = 'llm_api_key' 存系統 LLM API Key
SYSTEM_SETTING_LLM_KEY = "llm_api_key"
# System_Setting 表：key = 'deepgram_api_key' 存 Deepgram API Key（English System 音檔轉文字）
SYSTEM_SETTING_DEEPGRAM_KEY = "deepgram_api_key"


def get_llm_api_key() -> Optional[str]:
    """
    從 System_Setting 表取得系統唯一的 LLM API Key（key = 'llm_api_key' 的 value）。
    不需 person_id；若尚無資料或 value 為空，回傳 None。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", SYSTEM_SETTING_LLM_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        key = (resp.data[0].get("value") or "").strip()
        return key if key else None
    except Exception:
        return None


def get_deepgram_api_key() -> Optional[str]:
    """
    從 System_Setting 表取得 Deepgram API Key（key = 'deepgram_api_key' 的 value）。
    若尚無資料或 value 為空，回傳 None。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", SYSTEM_SETTING_DEEPGRAM_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        key = (resp.data[0].get("value") or "").strip()
        return key if key else None
    except Exception:
        return None


def get_llm_api_key_for_person(person_id: str) -> Optional[str]:
    """
    依 person_id 從 User 表取得該使用者的 LLM API Key。
    若 person_id 為空、找不到該使用者、或 llm_api_key 為空，回傳 None。
    """
    # 去除 person_id 前後空白
    pid = (person_id or "").strip()
    # 若為空，直接回傳 None
    if not pid:
        return None
    try:
        # 取得 Supabase 客戶端
        supabase = get_supabase()
        # 查詢 User 表中 person_id 符合的 llm_api_key
        resp = (
            supabase.table(USER_TABLE)
            .select("llm_api_key")
            .eq("person_id", pid)
            .limit(1)
            .execute()
        )
        # 若查無資料，回傳 None
        if not resp.data or len(resp.data) == 0:
            return None
        # 取得 llm_api_key 並去除空白，若為空則回傳 None
        key = (resp.data[0].get("llm_api_key") or "").strip()
        return key if key else None
    except Exception:
        return None
