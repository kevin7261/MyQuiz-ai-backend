"""
User／Course／Quiz Analysis 專屬 LLM API Key／模型（與 quiz-/rag-/exam- 出題設定分開）：
依 course_id 自 Course_Setting 讀取。
- User Analysis：key=user-analysis-api-key、user-analysis-llm-model
  （GET/PUT /v1/user-analyses/llm-api-key、/v1/user-analyses/llm-model）
- Course Analysis（RAG）：key=course-analysis-api-key
  （GET/PUT /v1/course-analyses/llm-api-key）
- Quiz Analysis：key=quiz-analysis-api-key、quiz-analysis-llm-model
  （GET/PUT /v1/quiz-analyses/llm-api-key、/v1/quiz-analyses/llm-model）
"""

from typing import Any, Optional

from services.bank_generation import BANK_QUIZ_LLM_MODEL
from utils.course_setting import (
    COURSE_SETTING_COURSE_ANALYSIS_API_KEY,
    COURSE_SETTING_QUIZ_ANALYSIS_API_KEY,
    COURSE_SETTING_QUIZ_ANALYSIS_LLM_MODEL,
    COURSE_SETTING_USER_ANALYSIS_API_KEY,
    COURSE_SETTING_USER_ANALYSIS_LLM_MODEL,
    fetch_course_setting_row,
    fetch_course_setting_value,
)


def get_user_analysis_api_key(course_id: int) -> Optional[str]:
    """User Analysis（POST /user-analyses/{id}/llm-analysis）。"""
    return fetch_course_setting_value(COURSE_SETTING_USER_ANALYSIS_API_KEY, course_id)


def user_analysis_api_key_exists(course_id: int) -> bool:
    value = get_user_analysis_api_key(course_id)
    return bool((value or "").strip())


def get_user_analysis_llm_model(course_id: int) -> str:
    """User Analysis LLM 模型；未設定時回傳 BANK_QUIZ_LLM_MODEL 預設。"""
    stored = fetch_course_setting_value(COURSE_SETTING_USER_ANALYSIS_LLM_MODEL, course_id)
    return stored or BANK_QUIZ_LLM_MODEL


def fetch_user_analysis_llm_model_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_USER_ANALYSIS_LLM_MODEL, course_id)


def get_course_analysis_api_key(course_id: int) -> Optional[str]:
    """Course Analysis（POST /course-analyses/{id}/llm-analysis）。"""
    return fetch_course_setting_value(COURSE_SETTING_COURSE_ANALYSIS_API_KEY, course_id)


def course_analysis_api_key_exists(course_id: int) -> bool:
    value = get_course_analysis_api_key(course_id)
    return bool((value or "").strip())


def get_quiz_analysis_api_key(course_id: int) -> Optional[str]:
    """Quiz Analysis（POST /quiz-analyses/{id}/llm-analysis）。"""
    return fetch_course_setting_value(COURSE_SETTING_QUIZ_ANALYSIS_API_KEY, course_id)


def quiz_analysis_api_key_exists(course_id: int) -> bool:
    value = get_quiz_analysis_api_key(course_id)
    return bool((value or "").strip())


def get_quiz_analysis_llm_model(course_id: int) -> str:
    """Quiz Analysis LLM 模型；未設定時回傳 BANK_QUIZ_LLM_MODEL 預設。"""
    stored = fetch_course_setting_value(COURSE_SETTING_QUIZ_ANALYSIS_LLM_MODEL, course_id)
    return stored or BANK_QUIZ_LLM_MODEL


def fetch_quiz_analysis_llm_model_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_QUIZ_ANALYSIS_LLM_MODEL, course_id)
