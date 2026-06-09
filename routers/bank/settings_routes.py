"""routers.bank LLM 設定 routes：bank 專屬 API Key／模型（與 rag 的 /v1/rag/llm-* 完全分開）。"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.openapi import openapi_body
from utils.supabase import get_supabase
from utils.course_setting import COURSE_SETTING_BANK_API_KEY, COURSE_SETTING_BANK_LLM_MODEL
from utils.bank_llm_key import (
    bank_api_key_exists,
    fetch_bank_api_key_setting_row,
    fetch_bank_llm_model_setting_row,
)
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)

_logger = logging.getLogger("routers.bank")

router = APIRouter(prefix="/bank", tags=["bank"])


class BankApiKeyExistsResponse(BaseModel):
    course_id: int
    exists: bool


class BankApiKeyResponse(BaseModel):
    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class BankLlmModelResponse(BaseModel):
    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutBankApiKeyRequest(BaseModel):
    api_key: str = Field("", description="Bank LLM API Key（寫入 Course_Setting key=bank-api-key）")


class PutBankLlmModelRequest(BaseModel):
    llm_model: str = Field("", description="Bank 出題／批改 LLM 模型（寫入 Course_Setting key=bank-llm-model）")


@router.get("/llm-api-key/exists", response_model=BankApiKeyExistsResponse, operation_id="bank_llm_api_key_exists")
def get_bank_api_key_exists(person_id: PersonId, course_id: CourseId):
    """查詢 Bank LLM API Key 是否已設定（Course_Setting key=bank-api-key）；不回傳 key 內容。"""
    _require_active_person(person_id)
    return BankApiKeyExistsResponse(course_id=course_id, exists=bank_api_key_exists(course_id))


@router.get("/llm-api-key", response_model=BankApiKeyResponse, operation_id="bank_get_llm_api_key")
def get_bank_api_key_setting(person_id: PersonId, course_id: CourseId):
    """讀取 Bank LLM API Key（Course_Setting key=bank-api-key）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_bank_api_key_setting_row(course_id)
    if not row:
        return BankApiKeyResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return BankApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=value or None)


@router.put("/llm-api-key", response_model=BankApiKeyResponse, operation_id="bank_put_llm_api_key")
def put_bank_api_key_setting(
    body: openapi_body(PutBankApiKeyRequest, {"api_key": "sk-..."}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 Bank LLM API Key（Course_Setting key=bank-api-key）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.api_key or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, COURSE_SETTING_BANK_API_KEY, value_to_save, course_id)
        if not row:
            return BankApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return BankApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=saved or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/llm-model", response_model=BankLlmModelResponse, operation_id="bank_get_llm_model")
def get_bank_llm_model_setting(person_id: PersonId, course_id: CourseId):
    """讀取 Bank 出題／批改 LLM 模型（Course_Setting key=bank-llm-model）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_bank_llm_model_setting_row(course_id)
    if not row:
        return BankLlmModelResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return BankLlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=value or None)


@router.put("/llm-model", response_model=BankLlmModelResponse, operation_id="bank_put_llm_model")
def put_bank_llm_model_setting(
    body: openapi_body(PutBankLlmModelRequest, {"llm_model": "gpt-5.4"}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 Bank 出題／批改 LLM 模型（Course_Setting key=bank-llm-model）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.llm_model or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, COURSE_SETTING_BANK_LLM_MODEL, value_to_save, course_id)
        if not row:
            return BankLlmModelResponse(course_id=course_id, llm_model=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return BankLlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=saved or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
