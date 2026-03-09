"""
LLM API Key 工具模組。
從 User 表依 person_id 取得個人 LLM API Key；或從 LLM_API_Key 表取得系統唯一 Key（Exam、course analysis 等使用）。
"""

# 引入 Optional 型別，表示可為 None
from typing import Optional

# 引入 Supabase 客戶端取得函數
from utils.supabase_client import get_supabase


def get_llm_api_key() -> Optional[str]:
    """
    從 LLM_API_Key 表取得系統唯一的 LLM API Key。
    表僅一筆，不需 person_id；若尚無資料或 key 為空，回傳 None。
    """
    try:
        # 取得 Supabase 客戶端
        supabase = get_supabase()
        # 查詢 LLM_API_Key 表，取最新一筆的 llm_api_key 欄位
        resp = (
            supabase.table("LLM_API_Key")
            .select("llm_api_key")
            .order("llm_api_key_id", desc=True)
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
        # 發生任何錯誤時回傳 None
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
            supabase.table("User")
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
