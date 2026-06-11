"""user-analyses／quiz-analyses 共用的分析指令 GET/PUT 端點（Course_Setting）。"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)
from utils.course_setting import fetch_course_setting_text
from utils.openapi import openapi_body
from utils.supabase import get_supabase


class AnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /{prefix}/analysis-user-prompt-text 回應（資料來自 Course_Setting）。"""

    course_id: int
    analysis_user_prompt_text: Optional[str] = None


class PutAnalysisUserPromptTextRequest(BaseModel):
    """PUT /{prefix}/analysis-user-prompt-text 的 body。"""

    analysis_user_prompt_text: str = Field(..., description="分析教師指令")


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
    )
    def get_analysis_user_prompt_text(person_id: PersonId, course_id: CourseId):
        f"""讀取分析指令（Course_Setting key={setting_key}，依 course_id）。"""
        _require_active_person(person_id)
        try:
            text = fetch_course_setting_text(setting_key, course_id)
            return AnalysisUserPromptTextResponse(
                course_id=int(course_id),
                analysis_user_prompt_text=text or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.put(
        "/analysis-user-prompt-text",
        response_model=AnalysisUserPromptTextResponse,
        summary="Update analysis_user_prompt_text",
        operation_id=f"{operation_id_prefix}_put_analysis_user_prompt_text",
    )
    def put_analysis_user_prompt_text(
        body: openapi_body(
            PutAnalysisUserPromptTextRequest,
            {"analysis_user_prompt_text": example},
        ),
        person_id: PersonId,
        course_id: CourseId,
    ):
        f"""寫入分析指令至 Course_Setting（key={setting_key}，依 course_id upsert；傳空字串可清除）。"""
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
            raise HTTPException(status_code=500, detail=str(e)) from e
