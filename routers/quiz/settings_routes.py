"""routers.quiz LLM 設定 routes：quiz 專屬 API Key／模型（與 bank-／exam-／rag- 完全分開）。"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.openapi import openapi_body
from utils.supabase import get_supabase
from utils.course_setting import COURSE_SETTING_QUIZ_API_KEY, COURSE_SETTING_QUIZ_LLM_MODEL
from utils.quiz_llm_key import (
    fetch_quiz_api_key_setting_row,
    fetch_quiz_llm_model_setting_row,
    quiz_api_key_exists,
)
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)

_logger = logging.getLogger("routers.quiz")

router = APIRouter(prefix="/quiz", tags=["quiz"])


class QuizApiKeyExistsResponse(BaseModel):
    course_id: int
    exists: bool


class QuizApiKeyResponse(BaseModel):
    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class QuizLlmModelResponse(BaseModel):
    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutQuizApiKeyRequest(BaseModel):
    api_key: str = Field("", description="Quiz LLM API Key（寫入 Course_Setting key=quiz-api-key）")


class PutQuizLlmModelRequest(BaseModel):
    llm_model: str = Field("", description="Quiz 出題／批改 LLM 模型（寫入 Course_Setting key=quiz-llm-model）")


@router.get("/llm-api-key/exists", response_model=QuizApiKeyExistsResponse, operation_id="quiz_llm_api_key_exists")
def get_quiz_api_key_exists(person_id: PersonId, course_id: CourseId):
    """查詢 Quiz LLM API Key 是否已設定（Course_Setting key=quiz-api-key）；不回傳 key 內容。"""
    _require_active_person(person_id)
    return QuizApiKeyExistsResponse(course_id=course_id, exists=quiz_api_key_exists(course_id))


@router.get("/llm-api-key", response_model=QuizApiKeyResponse, operation_id="quiz_get_llm_api_key")
def get_quiz_api_key_setting(person_id: PersonId, course_id: CourseId):
    """讀取 Quiz LLM API Key（Course_Setting key=quiz-api-key）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_quiz_api_key_setting_row(course_id)
    if not row:
        return QuizApiKeyResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return QuizApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=value or None)


@router.put("/llm-api-key", response_model=QuizApiKeyResponse, operation_id="quiz_put_llm_api_key")
def put_quiz_api_key_setting(
    body: openapi_body(PutQuizApiKeyRequest, {"api_key": "sk-..."}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 Quiz LLM API Key（Course_Setting key=quiz-api-key）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.api_key or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, COURSE_SETTING_QUIZ_API_KEY, value_to_save, course_id)
        if not row:
            return QuizApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return QuizApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=saved or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/llm-model", response_model=QuizLlmModelResponse, operation_id="quiz_get_llm_model")
def get_quiz_llm_model_setting(person_id: PersonId, course_id: CourseId):
    """讀取 Quiz 出題／批改 LLM 模型（Course_Setting key=quiz-llm-model）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_quiz_llm_model_setting_row(course_id)
    if not row:
        return QuizLlmModelResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return QuizLlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=value or None)


@router.put("/llm-model", response_model=QuizLlmModelResponse, operation_id="quiz_put_llm_model")
def put_quiz_llm_model_setting(
    body: openapi_body(PutQuizLlmModelRequest, {"llm_model": "gpt-5.4"}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 Quiz 出題／批改 LLM 模型（Course_Setting key=quiz-llm-model）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.llm_model or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, COURSE_SETTING_QUIZ_LLM_MODEL, value_to_save, course_id)
        if not row:
            return QuizLlmModelResponse(course_id=course_id, llm_model=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return QuizLlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=saved or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
