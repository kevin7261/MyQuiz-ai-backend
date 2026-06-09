"""
Bank 專屬 LLM API Key／模型（與 rag 完全分開）：依 course_id 自 Course_Setting 讀取。
- Bank Key：key=bank-api-key（GET/PUT /v1/bank/llm-api-key；GET /v1/bank/llm-api-key/exists）
- Bank 模型（出題、批改共用）：key=bank-llm-model（GET/PUT /v1/bank/llm-model；未設定 fallback BANK_QUIZ_LLM_MODEL）
"""

from typing import Any, Optional

from services.bank_generation import BANK_QUIZ_LLM_MODEL
from utils.course_setting import (
    COURSE_SETTING_BANK_API_KEY,
    COURSE_SETTING_BANK_LLM_MODEL,
    fetch_course_setting_row,
    fetch_course_setting_value,
)


def get_bank_api_key(course_id: int) -> Optional[str]:
    """Bank 出題／批改／建 FAISS 用 API Key。"""
    return fetch_course_setting_value(COURSE_SETTING_BANK_API_KEY, course_id)


def get_bank_llm_model(course_id: int) -> str:
    """Bank 出題／批改 LLM 模型；未設定時回傳 BANK_QUIZ_LLM_MODEL 預設。"""
    stored = fetch_course_setting_value(COURSE_SETTING_BANK_LLM_MODEL, course_id)
    return stored or BANK_QUIZ_LLM_MODEL


def fetch_bank_api_key_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_BANK_API_KEY, course_id)


def fetch_bank_llm_model_setting_row(course_id: int) -> Optional[dict[str, Any]]:
    return fetch_course_setting_row(COURSE_SETTING_BANK_LLM_MODEL, course_id)


def bank_api_key_exists(course_id: int) -> bool:
    value = fetch_course_setting_value(COURSE_SETTING_BANK_API_KEY, course_id)
    return bool((value or "").strip())
