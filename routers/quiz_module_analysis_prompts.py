"""
Quiz 模組弱點分析指令（Course_Setting）GET/PUT 端點共用實作。

掛載於：
- /v1/quiz/user-analysis-user-prompt-text、/v1/quiz/quiz-analysis-user-prompt-text
- /v1/user-analyses/user-analysis-user-prompt-text（須在 /{user_analysis_id} 之前）
- /v1/quiz-analyses/quiz-analysis-user-prompt-text（須在 /{quiz_analysis_id} 之前）
"""

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
from utils.course_setting import (
    COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT_KEY,
    COURSE_SETTING_USER_ANALYSIS_USER_PROMPT_TEXT_KEY,
    fetch_course_setting_text,
)
from utils.openapi import openapi_body
from utils.supabase import get_supabase


class UserAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT user-analysis-user-prompt-text 回應。"""

    course_id: Optional[int] = None
    user_analysis_user_prompt_text: Optional[str] = None


class PutUserAnalysisUserPromptTextRequest(BaseModel):
    user_analysis_user_prompt_text: str = Field(
        ..., description="個人弱點分析使用者 Prompt 文字"
    )


class QuizAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT quiz-analysis-user-prompt-text 回應。"""

    course_id: Optional[int] = None
    quiz_analysis_user_prompt_text: Optional[str] = None


class PutQuizAnalysisUserPromptTextRequest(BaseModel):
    quiz_analysis_user_prompt_text: str = Field(
        ..., description="測驗課程分析使用者 Prompt 文字"
    )


def register_user_analysis_prompt_routes(
    router: APIRouter,
    *,
    get_operation_id: str = "get_user_analysis_user_prompt_text",
    put_operation_id: str = "put_user_analysis_user_prompt_text",
) -> None:
    """註冊 /user-analysis-user-prompt-text GET/PUT。"""

    @router.get(
        "/user-analysis-user-prompt-text",
        response_model=UserAnalysisUserPromptTextResponse,
        operation_id=get_operation_id,
    )
    def get_user_analysis_user_prompt_text_setting(person_id: PersonId, course_id: CourseId):
        """取得個人弱點分析指令（Course_Setting key=user_analysis_user_prompt_text，依 course_id）。"""
        _require_active_person(person_id)
        try:
            text = fetch_course_setting_text(
                COURSE_SETTING_USER_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
            )
            return UserAnalysisUserPromptTextResponse(
                course_id=course_id,
                user_analysis_user_prompt_text=text or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.put(
        "/user-analysis-user-prompt-text",
        response_model=UserAnalysisUserPromptTextResponse,
        operation_id=put_operation_id,
    )
    def put_user_analysis_user_prompt_text_setting(
        body: openapi_body(
            PutUserAnalysisUserPromptTextRequest,
            {"user_analysis_user_prompt_text": "string"},
        ),
        person_id: PersonId,
        course_id: CourseId,
    ):
        """寫入個人弱點分析指令至 Course_Setting（依 course_id upsert；傳空字串可清除）。"""
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.user_analysis_user_prompt_text or "").strip()
        try:
            row = _upsert_setting_and_get_row(
                get_supabase(),
                COURSE_SETTING_USER_ANALYSIS_USER_PROMPT_TEXT_KEY,
                value_to_save,
                course_id,
            )
            if not row:
                raise HTTPException(status_code=500, detail="寫入 Course_Setting 失敗")
            return UserAnalysisUserPromptTextResponse(
                course_id=course_id,
                user_analysis_user_prompt_text=value_to_save or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


def register_quiz_analysis_prompt_routes(
    router: APIRouter,
    *,
    get_operation_id: str = "get_quiz_analysis_user_prompt_text",
    put_operation_id: str = "put_quiz_analysis_user_prompt_text",
) -> None:
    """註冊 /quiz-analysis-user-prompt-text GET/PUT。"""

    @router.get(
        "/quiz-analysis-user-prompt-text",
        response_model=QuizAnalysisUserPromptTextResponse,
        operation_id=get_operation_id,
    )
    def get_quiz_analysis_user_prompt_text_setting(person_id: PersonId, course_id: CourseId):
        """取得測驗課程分析指令（Course_Setting key=quiz_analysis_user_prompt_text，依 course_id）。"""
        _require_active_person(person_id)
        try:
            text = fetch_course_setting_text(
                COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
            )
            return QuizAnalysisUserPromptTextResponse(
                course_id=course_id,
                quiz_analysis_user_prompt_text=text or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.put(
        "/quiz-analysis-user-prompt-text",
        response_model=QuizAnalysisUserPromptTextResponse,
        operation_id=put_operation_id,
    )
    def put_quiz_analysis_user_prompt_text_setting(
        body: openapi_body(
            PutQuizAnalysisUserPromptTextRequest,
            {"quiz_analysis_user_prompt_text": "string"},
        ),
        person_id: PersonId,
        course_id: CourseId,
    ):
        """寫入測驗課程分析指令至 Course_Setting（依 course_id upsert；傳空字串可清除）。"""
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.quiz_analysis_user_prompt_text or "").strip()
        try:
            row = _upsert_setting_and_get_row(
                get_supabase(),
                COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT_KEY,
                value_to_save,
                course_id,
            )
            if not row:
                raise HTTPException(status_code=500, detail="寫入 Course_Setting 失敗")
            return QuizAnalysisUserPromptTextResponse(
                course_id=course_id,
                quiz_analysis_user_prompt_text=value_to_save or None,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


def register_quiz_module_analysis_prompt_routes(router: APIRouter) -> None:
    """在 /quiz 設定 router 上註冊兩組分析指令端點。"""
    register_user_analysis_prompt_routes(
        router,
        get_operation_id="quiz_get_user_analysis_user_prompt_text",
        put_operation_id="quiz_put_user_analysis_user_prompt_text",
    )
    register_quiz_analysis_prompt_routes(
        router,
        get_operation_id="quiz_get_quiz_analysis_user_prompt_text",
        put_operation_id="quiz_put_quiz_analysis_user_prompt_text",
    )
