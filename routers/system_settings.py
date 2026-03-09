"""
系統設定相關 API 模組。
提供 LLM API Key 的 GET/PUT，供 Exam、course analysis 等使用。
"""

# 引入 Optional 型別
from typing import Optional

# 引入 FastAPI 與 HTTPException
from fastapi import APIRouter, HTTPException
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 引入 UTC 時間工具
from utils.datetime_utils import now_utc_iso
# 引入 Supabase 客戶端
from utils.supabase_client import get_supabase

# 建立路由，前綴為 /system-settings，標籤為 system-settings
router = APIRouter(prefix="/system-settings", tags=["system-settings"])

# LLM_API_Key 表查詢欄位
LLM_API_KEY_COLUMNS = "llm_api_key_id, llm_api_key, updated_at, created_at"


class LlmApiKeyResponse(BaseModel):
    """GET /system-settings/llm-api-key 回應。表僅一筆，直接取用，不需 person_id。"""
    llm_api_key_id: Optional[int] = None
    llm_api_key: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class PutLlmApiKeyRequest(BaseModel):
    """PUT /system-settings/llm-api-key 請求：僅 llm_api_key（表僅一筆，無 person_id）。"""
    llm_api_key: str = Field(..., description="LLM API Key，可為空字串表示清除")


@router.get("/llm-api-key", response_model=LlmApiKeyResponse)
def get_llm_api_key():
    """
    取得系統唯一的 LLM API Key。
    資料表僅一筆，直接取用，不需傳 person_id。
    若尚無資料，回傳 llm_api_key_id 等皆為 null。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select(LLM_API_KEY_COLUMNS)
            .order("llm_api_key_id", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return LlmApiKeyResponse()
        row = resp.data[0]
        return LlmApiKeyResponse(
            llm_api_key_id=row.get("llm_api_key_id"),
            llm_api_key=row.get("llm_api_key"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-api-key", response_model=LlmApiKeyResponse)
def put_llm_api_key(body: PutLlmApiKeyRequest):
    """
    寫入或更新系統唯一的 LLM API Key。
    表僅一筆：已有則更新，否則新增。llm_api_key 可傳空字串表示清除。
    """
    key_value = (body.llm_api_key or "").strip() or None
    now = now_utc_iso()
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select("llm_api_key_id")
            .order("llm_api_key_id", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data and len(resp.data) > 0:
            row_id = resp.data[0].get("llm_api_key_id")
            supabase.table("LLM_API_Key").update({
                "llm_api_key": key_value,
                "updated_at": now,
            }).eq("llm_api_key_id", row_id).execute()
        else:
            supabase.table("LLM_API_Key").insert({
                "llm_api_key": key_value,
                "updated_at": now,
                "created_at": now,
            }).execute()
        resp2 = (
            supabase.table("LLM_API_Key")
            .select(LLM_API_KEY_COLUMNS)
            .order("llm_api_key_id", desc=True)
            .limit(1)
            .execute()
        )
        if not resp2.data or len(resp2.data) == 0:
            return LlmApiKeyResponse()
        row = resp2.data[0]
        return LlmApiKeyResponse(
            llm_api_key_id=row.get("llm_api_key_id"),
            llm_api_key=row.get("llm_api_key"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
