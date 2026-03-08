"""LLM API Key 工具：從 LLM_API_Key 表取得唯一一筆 API Key（表僅一筆，不依 person_id）。"""

from typing import Optional

from utils.supabase_client import get_supabase


def get_llm_api_key() -> Optional[str]:
    """
    從 LLM_API_Key 表取得系統唯一的 LLM API Key。表僅一筆，不需 person_id。
    若尚無資料或 key 為空，回傳 None。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select("llm_api_key")
            .order("llm_api_key_id", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        key = (resp.data[0].get("llm_api_key") or "").strip()
        return key if key else None
    except Exception:
        return None


def get_llm_api_key_for_person(person_id: str) -> Optional[str]:
    """
    取得系統唯一的 LLM API Key。表僅一筆，person_id 僅保留相容用，實際不參與查詢。
    """
    return get_llm_api_key()
