"""
LLM API Key：僅從環境變數 LLM_API_KEY 或 OPENAI_API_KEY 讀取（main.py load_dotenv 載入 .env）。
依使用者：先 User.llm_api_key，若為空則同上環境變數。
"""

import os
from typing import Optional

from utils.db_tables import USER_TABLE
from utils.supabase_client import get_supabase


def _llm_api_key_from_env() -> Optional[str]:
    """優先 LLM_API_KEY，其次 OPENAI_API_KEY。"""
    for name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        k = (os.environ.get(name) or "").strip()
        if k:
            return k
    return None


def get_llm_api_key() -> Optional[str]:
    """系統 LLM API Key：僅環境變數 LLM_API_KEY 或 OPENAI_API_KEY。"""
    return _llm_api_key_from_env()


def get_llm_api_key_for_person(person_id: str) -> Optional[str]:
    """
    依 person_id 從 User 表取得 llm_api_key；欄位為空時改讀環境變數 LLM_API_KEY／OPENAI_API_KEY。
    person_id 為空或查無使用者列時回傳 None（不套用 env，避免錯 person_id 仍帶到金鑰）。
    """
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("llm_api_key")
            .eq("person_id", pid)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        key = (resp.data[0].get("llm_api_key") or "").strip()
        if key:
            return key
        return _llm_api_key_from_env()
    except Exception:
        return None
