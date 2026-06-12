"""
LLM API Key／出題模型：依 course_id 自 Course_Setting 讀取。
- RAG Key：key=rag-api-key（GET/PUT /v1/rag/llm-api-key；GET /v1/rag/llm-api-key/exists 查是否已設定）
- LLM 模型（出題、批改、弱點分析共用）：key=llm-model（GET/PUT /v1/rag/llm-model；未設定時 fallback `QUIZ_LLM_MODEL`）
- Exam Key：key=exam-api-key（GET/PUT /v1/exam/llm-api-key；GET /v1/exam/llm-api-key/exists 查是否已設定；個人弱點分析）
- 課程弱點分析：沿用 RAG Key（key=rag-api-key）；無專屬金鑰
- User Analysis（Quiz 個人弱點）：key=user-analysis-api-key（GET/PUT /v1/user-analyses/llm-api-key）
- Quiz Analysis（Quiz 課程弱點）：key=quiz-analysis-api-key（GET/PUT /v1/quiz-analyses/llm-api-key）
"""

from typing import Any, Optional

from services.quiz_generation import QUIZ_LLM_MODEL
from utils.course_setting import (
    COURSE_SETTING_EXAM_API_KEY,
    COURSE_SETTING_RAG_API_KEY,
    COURSE_SETTING_LLM_MODEL,
    fetch_course_setting_row,
    fetch_course_setting_value,
)


def get_rag_api_key(course_id: int) -> Optional[str]:
    """RAG 出題／批改／建 FAISS 等 /rag 路徑用。"""
    return fetch_course_setting_value(COURSE_SETTING_RAG_API_KEY, course_id)


def get_rag_llm_model(course_id: int) -> str:
    """RAG／Exam 出題、批改與弱點分析 LLM 模型；Course_Setting 未設定時回傳 `QUIZ_LLM_MODEL` 預設值。"""
    stored = fetch_course_setting_value(COURSE_SETTING_LLM_MODEL, course_id)
    return stored or QUIZ_LLM_MODEL


def get_exam_api_key(course_id: int) -> Optional[str]:
    """Exam 出題／批改、個人弱點分析（Course_Setting key=exam-api-key）。"""
    return fetch_course_setting_value(COURSE_SETTING_EXAM_API_KEY, course_id)


def get_person_analysis_api_key(course_id: int) -> Optional[str]:
    """個人弱點分析（POST /person-analysis/llm-analysis）：exam-api-key。"""
    return get_exam_api_key(course_id)


def fetch_api_key_setting_row(key: str, course_id: int) -> Optional[dict[str, Any]]:
    """讀取 Course_Setting 整列（course_id + key）；無列時回傳 None。"""
    return fetch_course_setting_row(key, course_id)


def course_api_key_exists(key: str, course_id: int) -> bool:
    """Course_Setting 是否已有非空 API Key（value 經 strip 後非空）。"""
    value = fetch_course_setting_value(key, course_id)
    return bool((value or "").strip())
