"""系統設定相關 API：LLM API Key 等。"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from utils.datetime_utils import now_utc_iso
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/system-settings", tags=["system-settings"])

LLM_API_KEY_COLUMNS = "llm_api_key_id, person_id, llm_api_key, updated_at, created_at"


class LlmApiKeyResponse(BaseModel):
    """GET /system-settings/llm-api-key 回應。取時不需傳 person_id（依認證或 Header）。"""
    llm_api_key_id: Optional[int] = None
    llm_api_key: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class PutLlmApiKeyRequest(BaseModel):
    """PUT /system-settings/llm-api-key 請求：person_id 與 llm_api_key 必填。"""
    person_id: str = Field(..., description="使用者識別")
    llm_api_key: str = Field(..., description="LLM API Key，可為空字串表示清除")


@router.get("/llm-api-key", response_model=LlmApiKeyResponse)
def get_llm_api_key(
    x_person_id: Optional[str] = Header(None, alias="X-Person-Id", description="選填，若有認證可省略"),
):
    """
    取得目前使用者的 LLM API Key。取時可不傳 person_id（若後端有認證則依認證）；無認證時請傳 Header X-Person-Id。
    若該 person_id 尚無資料，回傳 llm_api_key_id 等皆為 null。
    """
    person_id = (x_person_id or "").strip()
    if not person_id:
        return LlmApiKeyResponse()
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select(LLM_API_KEY_COLUMNS)
            .eq("person_id", person_id)
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
    寫入或更新 LLM API Key。輸入需傳 person_id（body 必填）。
    若該 person_id 已有資料則更新，否則新增一筆。llm_api_key 可傳空字串表示清除。
    """
    person_id = (body.person_id or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    key_value = (body.llm_api_key or "").strip() or None
    now = now_utc_iso()
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("LLM_API_Key")
            .select("llm_api_key_id")
            .eq("person_id", person_id)
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
                "person_id": person_id,
                "llm_api_key": key_value,
                "updated_at": now,
                "created_at": now,
            }).execute()
        resp2 = (
            supabase.table("LLM_API_Key")
            .select(LLM_API_KEY_COLUMNS)
            .eq("person_id", person_id)
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
