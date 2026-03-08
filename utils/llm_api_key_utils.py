"""LLM API Key 工具：依 person_id 從 LLM_API_Key 表取得 API Key。"""

from typing import Optional

from utils.supabase_client import get_supabase


def get_llm_api_key_for_person(person_id: str) -> Optional[str]:
    """
    依 person_id 從 LLM_API_Key 表取得該使用者的 LLM API Key。
    若該 person_id 尚無資料或 key 為空，回傳 None。
    """
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select("llm_api_key")
            .eq("person_id", pid)
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
