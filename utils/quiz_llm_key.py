"""
Quiz（測驗／Test，搭配 bank 出題）專屬 LLM API Key／模型（與 bank-/exam-/rag- 完全分開）：
依 course_id 自 Course_Setting 讀取。
- Quiz Key：key=quiz-api-key（GET/PUT /v1/quiz/llm-api-key；GET /v1/quiz/llm-api-key/exists）
- Quiz 模型（出題、批改共用）：key=quiz-llm-model（GET/PUT /v1/quiz/llm-model；未設定 fallback BANK_QUIZ_LLM_MODEL）

出題／批改沿用 bank 的 LLM 管線（services.bank_generation／services.bank_answering），但金鑰與模型
由此模組的 quiz- 設定提供，與 bank 的 bank- 設定互不影響。
"""

from typing import Any, Optional

from services.bank_generation import BANK_QUIZ_LLM_MODEL
from utils.course_setting import (
    COURSE_SETTING_QUIZ_API_KEY,
    COURSE_SETTING_QUIZ_LLM_MODEL,
    fetch_course_setting_row,
    fetch_course_setting_value,
)


def get_quiz_api_key(course_id: int) -> Optional[str]:
    """Quiz 出題／批改／建 FAISS 用 API Key。"""
    return fetch_course_setting_value(COURSE_SETTING_QUIZ_API_KEY, course_id)


def get_quiz_llm_model(course_id: int) -> str:
    """Quiz 出題／批改 LLM 模型；未設定時回傳 BANK_QUIZ_LLM_MODEL 預設。"""
    stored = fetch_course_setting_value(COURSE_SETTING_QUIZ_LLM_MODEL, course_id)
    return stored or BANK_QUIZ_LLM_MODEL


def fetch_quiz_api_key_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_QUIZ_API_KEY, course_id)


def fetch_quiz_llm_model_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_QUIZ_LLM_MODEL, course_id)


def quiz_api_key_exists(course_id: int) -> bool:
    value = fetch_course_setting_value(COURSE_SETTING_QUIZ_API_KEY, course_id)
    return bool((value or "").strip())
