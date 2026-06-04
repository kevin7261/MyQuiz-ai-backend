"""
LLM API Key／出題模型：依 course_id 自 Course_Setting 讀取。
- RAG Key：key=rag-api-key（GET/PUT /rag/llm_api_key）
- LLM 模型（出題、批改、弱點分析共用）：key=llm-model（GET/PUT /rag/llm_model；未設定時 fallback `QUIZ_LLM_MODEL`）
- Exam Key：key=exam-api-key（GET/PUT /exam/llm_api_key）
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
    """Exam 出題／批改、個人／課程弱點分析（Course_Setting key=exam-api-key）。"""
    return fetch_course_setting_value(COURSE_SETTING_EXAM_API_KEY, course_id)


def get_person_analysis_api_key(course_id: int) -> Optional[str]:
    """個人弱點分析 LLM：僅 Course_Setting key=exam-api-key。"""
    return get_exam_api_key(course_id)


def get_weakness_analysis_api_key(course_id: int) -> Optional[str]:
    """課程弱點分析：僅 exam-api-key（與個人分析相同）。"""
    return get_exam_api_key(course_id)


def fetch_api_key_setting_row(key: str, course_id: int) -> Optional[dict[str, Any]]:
    """讀取 Course_Setting 整列（course_id + key）；無列時回傳 None。"""
    return fetch_course_setting_row(key, course_id)
