"""
LLM API Key：僅依 course_id 自 Course_Setting 讀取。
- RAG：key=rag-api-key（GET/PUT /rag/api_key）
- Exam：key=exam-api-key（GET/PUT /exam/api_key）
"""

from typing import Any, Optional

from utils.course_setting import (
    COURSE_SETTING_EXAM_API_KEY,
    COURSE_SETTING_RAG_API_KEY,
    fetch_course_setting_row,
    fetch_course_setting_value,
)


def get_rag_api_key(course_id: int) -> Optional[str]:
    """RAG 出題／批改／建 FAISS 等 /rag 路徑用。"""
    return fetch_course_setting_value(COURSE_SETTING_RAG_API_KEY, course_id)


def get_exam_api_key(course_id: int) -> Optional[str]:
    """Exam 出題／批改及 exam 弱點分析用。"""
    return fetch_course_setting_value(COURSE_SETTING_EXAM_API_KEY, course_id)


def fetch_api_key_setting_row(key: str, course_id: int) -> Optional[dict[str, Any]]:
    """讀取 Course_Setting 整列（course_id + key）；無列時回傳 None。"""
    return fetch_course_setting_row(key, course_id)
