"""user-analyses／quiz-analyses／course-analyses 共用的分析設定 GET/PUT 端點（Course_Setting）。"""

import logging
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import CurrentUser, PersonId
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)
from utils.course_setting import fetch_course_setting_text
from utils.llm_key import fetch_api_key_setting_row
from utils.openapi import openapi_body
from utils.supabase import get_supabase

_logger = logging.getLogger(__name__)


class AnalysisApiKeyExistsResponse(BaseModel):
    """GET /{prefix}/llm-api-key/exists 回應。"""

    course_id: int
    exists: bool = Field(..., description="該課程是否已設定非空 API Key")


class AnalysisApiKeyResponse(BaseModel):
    """GET/PUT /{prefix}/llm-api-key 回應。"""

    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class PutAnalysisApiKeyRequest(BaseModel):
    """PUT /{prefix}/llm-api-key 的 body。"""

    api_key: str = Field(..., description="LLM API Key")


class AnalysisLlmModelResponse(BaseModel):
    """GET/PUT /{prefix}/llm-model 回應。"""

    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutAnalysisLlmModelRequest(BaseModel):
    """PUT /{prefix}/llm-model 的 body。"""

    llm_model: str = Field(..., description="弱點分析 LLM 模型")


class AnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /{prefix}/analysis-user-prompt-text 回應（資料來自 Course_Setting）。"""

    course_id: int
    analysis_user_prompt_text: Optional[str] = ""


class PutAnalysisUserPromptTextRequest(BaseModel):
    """PUT /{prefix}/analysis-user-prompt-text 的 body。"""

    analysis_user_prompt_text: str = Field("", description="分析教師指令")


def register_analysis_llm_api_key_routes(
    router: APIRouter,
    *,
    setting_key: str,
    operation_id_prefix: str,
    title: str,
    api_key_exists: Callable[[int], bool],
) -> None:
    """在 user-analyses 或 quiz-analyses router 註冊 GET/PUT /llm-api-key（+ /exists）。"""

    @router.get(
        "/llm-api-key/exists",
        response_model=AnalysisApiKeyExistsResponse,
        summary="Get llm_api_key exists",
        operation_id=f"{operation_id_prefix}_llm_api_key_exists",
        description=f"查詢 {title} LLM API Key 是否已設定（Course_Setting key={setting_key}）；不回傳 key 內容。",
    )
    def get_analysis_api_key_exists(caller: CurrentUser, course_id: CourseId):
        _require_active_person(caller.person_id, caller.college_id)
        return AnalysisApiKeyExistsResponse(
            course_id=int(course_id),
            exists=api_key_exists(int(course_id)),
        )

    @router.get(
        "/llm-api-key",
        response_model=AnalysisApiKeyResponse,
        summary="Get llm_api_key",
        operation_id=f"{operation_id_prefix}_get_llm_api_key",
        description=f"讀取 {title} LLM API Key（Course_Setting key={setting_key}，依 course_id）。",
    )
    def get_analysis_api_key_setting(person_id: PersonId, course_id: CourseId):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        row = fetch_api_key_setting_row(setting_key, course_id)
        if not row:
            return AnalysisApiKeyResponse(course_id=int(course_id))
        value = (row.get("value") or "").strip()
        return AnalysisApiKeyResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=int(course_id),
            api_key=value or None,
        )

    @router.put(
        "/llm-api-key",
        response_model=AnalysisApiKeyResponse,
        summary="Put llm_api_key",
        operation_id=f"{operation_id_prefix}_put_llm_api_key",
        description=f"寫入 {title} LLM API Key（Course_Setting key={setting_key}，依 course_id）。",
    )
    def put_analysis_api_key_setting(
        body: openapi_body(PutAnalysisApiKeyRequest, {"api_key": "sk-..."}),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.api_key or "").strip()
        try:
            row = _upsert_setting_and_get_row(
                get_supabase(), setting_key, value_to_save, course_id
            )
            if not row:
                return AnalysisApiKeyResponse(
                    course_id=int(course_id),
                    api_key=value_to_save or None,
                )
            saved = (row.get("value") or "").strip()
            return AnalysisApiKeyResponse(
                course_setting_id=row.get("course_setting_id"),
                course_id=int(course_id),
                api_key=saved or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            _logger.exception("PUT /llm-api-key 失敗")
            raise HTTPException(
                status_code=500, detail="儲存失敗，請稍後再試"
            ) from e


def register_analysis_llm_model_routes(
    router: APIRouter,
    *,
    setting_key: str,
    operation_id_prefix: str,
    title: str,
    fetch_llm_model_setting_row: Callable[[int], Optional[dict]],
) -> None:
    """在 user-analyses 或 quiz-analyses router 註冊 GET/PUT /llm-model。"""

    @router.get(
        "/llm-model",
        response_model=AnalysisLlmModelResponse,
        summary="Get llm_model",
        operation_id=f"{operation_id_prefix}_get_llm_model",
        description=f"讀取 {title} LLM 模型（Course_Setting key={setting_key}，依 course_id）。",
    )
    def get_analysis_llm_model_setting(person_id: PersonId, course_id: CourseId):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        row = fetch_llm_model_setting_row(int(course_id))
        if not row:
            return AnalysisLlmModelResponse(course_id=int(course_id))
        value = (row.get("value") or "").strip()
        return AnalysisLlmModelResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=int(course_id),
            llm_model=value or None,
        )

    @router.put(
        "/llm-model",
        response_model=AnalysisLlmModelResponse,
        summary="Put llm_model",
        operation_id=f"{operation_id_prefix}_put_llm_model",
        description=f"寫入 {title} LLM 模型（Course_Setting key={setting_key}，依 course_id）。",
    )
    def put_analysis_llm_model_setting(
        body: openapi_body(PutAnalysisLlmModelRequest, {"llm_model": "gpt-5.4"}),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.llm_model or "").strip()
        try:
            row = _upsert_setting_and_get_row(
                get_supabase(), setting_key, value_to_save, course_id
            )
            if not row:
                return AnalysisLlmModelResponse(
                    course_id=int(course_id),
                    llm_model=value_to_save or None,
                )
            saved = (row.get("value") or "").strip()
            return AnalysisLlmModelResponse(
                course_setting_id=row.get("course_setting_id"),
                course_id=int(course_id),
                llm_model=saved or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            _logger.exception("PUT /llm-model 失敗")
            raise HTTPException(
                status_code=500, detail="儲存失敗，請稍後再試"
            ) from e


def register_analysis_user_prompt_routes(
    router: APIRouter,
    *,
    setting_key: str,
    operation_id_prefix: str,
    example: str,
) -> None:
    """在 user-analyses 或 quiz-analyses router 註冊 GET/PUT /analysis-user-prompt-text。"""

    @router.get(
        "/analysis-user-prompt-text",
        response_model=AnalysisUserPromptTextResponse,
        summary="Get analysis_user_prompt_text",
        operation_id=f"{operation_id_prefix}_get_analysis_user_prompt_text",
        description=f"讀取分析指令（Course_Setting key={setting_key}，依 course_id）。",
    )
    def get_analysis_user_prompt_text(caller: CurrentUser, course_id: CourseId):
        _require_active_person(caller.person_id, caller.college_id)
        try:
            text = fetch_course_setting_text(setting_key, course_id)
            return AnalysisUserPromptTextResponse(
                course_id=int(course_id),
                analysis_user_prompt_text=text or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            _logger.exception("GET /analysis-user-prompt-text 失敗")
            raise HTTPException(
                status_code=500, detail="讀取失敗，請稍後再試"
            ) from e

    @router.put(
        "/analysis-user-prompt-text",
        response_model=AnalysisUserPromptTextResponse,
        summary="Update analysis_user_prompt_text",
        operation_id=f"{operation_id_prefix}_put_analysis_user_prompt_text",
        description=f"寫入分析指令至 Course_Setting（key={setting_key}，依 course_id upsert；傳空字串可清除）。",
    )
    def put_analysis_user_prompt_text(
        body: openapi_body(
            PutAnalysisUserPromptTextRequest,
            {"analysis_user_prompt_text": example},
        ),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.analysis_user_prompt_text or "").strip()
        try:
            row = _upsert_setting_and_get_row(
                get_supabase(), setting_key, value_to_save, course_id
            )
            if not row:
                raise HTTPException(status_code=500, detail="寫入 Course_Setting 失敗")
            return AnalysisUserPromptTextResponse(
                course_id=int(course_id),
                analysis_user_prompt_text=value_to_save or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            _logger.exception("PUT /analysis-user-prompt-text 失敗")
            raise HTTPException(
                status_code=500, detail="儲存失敗，請稍後再試"
            ) from e
