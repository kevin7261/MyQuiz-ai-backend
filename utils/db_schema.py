"""
與 Supabase／Postgres 實際 schema 一致的表名常數。
若遷移或重命名表，僅需改此處。
"""

from __future__ import annotations

from typing import Any

# public."User"
USER_TABLE = "User"
# enrollments / per-course profile（user_type、llm_api_key）
USER_COURSE_RELATION_TABLE = "User_Course_Relation"

# Supabase PostgREST：deleted 為 false 或 null 視為有效列（與舊資料相容）
ACTIVE_DELETED_FILTER = "deleted.eq.false,deleted.is.null"

# RAG 相關表名
RAG_TABLE = "Rag"
RAG_UNIT_TABLE = "Rag_Unit"
RAG_QUIZ_TABLE = "Rag_Quiz"

# 三表共用 course_id（bigint null default 0）
RAG_COURSE_ID_DEFAULT = 0

# SELECT 欄位順序同 public DDL（rag_tab_id → person_id → course_id → …）
RAG_SELECT_COLUMNS = (
    "rag_id, rag_tab_id, person_id, course_id, tab_name, file_size, file_metadata, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_NO_FILE_METADATA = (
    "rag_id, rag_tab_id, person_id, course_id, tab_name, file_size, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_LEGACY = (
    "rag_id, rag_tab_id, person_id, tab_name, file_size, file_metadata, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA = (
    "rag_id, rag_tab_id, person_id, tab_name, file_size, "
    "local, deleted, updated_at, created_at"
)

RAG_UNIT_SELECT_COLUMNS = (
    "rag_unit_id, rag_tab_id, person_id, course_id, unit_name, folder_combination, unit_type, "
    "repack_file_name, rag_file_name, rag_file_size, rag_chunk_size, rag_chunk_overlap, "
    "transcript, text_file_name, mp3_file_name, youtube_url, deleted, updated_at, created_at"
)

RAG_QUIZ_SELECT_COLUMNS = (
    "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, for_exam, deleted, "
    "updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, for_exam, deleted, "
    "updated_at, created_at"
)


def rag_quiz_list_row(row: dict[str, Any]) -> dict[str, Any]:
    """GET /rag/tabs、GET /rag/tab/units 之 quizzes[] 單筆（欄位順序對齊 public.Rag_Quiz）。"""
    ans = row.get("answer_content")
    if ans is None:
        ans = row.get("quiz_answer")
    ans_s = (ans or "") if ans is not None else ""
    return {
        "rag_quiz_id": row.get("rag_quiz_id"),
        "rag_tab_id": row.get("rag_tab_id") or "",
        "rag_unit_id": row.get("rag_unit_id"),
        "person_id": row.get("person_id") or "",
        "course_id": row.get("course_id"),
        "follow_up": bool(row.get("follow_up")),
        "quiz_name": row.get("quiz_name") or "",
        "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
        "quiz_content": row.get("quiz_content"),
        "quiz_hint": row.get("quiz_hint"),
        "quiz_answer_reference": row.get("quiz_answer_reference"),
        "answer_user_prompt_text": row.get("answer_user_prompt_text"),
        "answer_content": ans_s,
        "quiz_answer": ans_s,
        "answer_critique": row.get("answer_critique"),
        "for_exam": row.get("for_exam"),
        "deleted": row.get("deleted"),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
    }


# Exam 相關表名
EXAM_TABLE = "Exam"
EXAM_QUIZ_TABLE = "Exam_Quiz"

EXAM_COURSE_ID_DEFAULT = 0

EXAM_SELECT_COLUMNS = (
    "exam_id, exam_tab_id, person_id, course_id, tab_name, local, deleted, updated_at, created_at"
)

EXAM_QUIZ_SELECT_COLUMNS = (
    "exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID = (
    "exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, "
    "updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "quiz_rate, answer_user_prompt_text, answer_content, answer_critique, updated_at, created_at"
)


def exam_quiz_list_row(row: dict[str, Any]) -> dict[str, Any]:
    """GET /exam/tabs 等之 quizzes[] 單筆（欄位順序對齊 public.Exam_Quiz）。"""
    ans = row.get("answer_content")
    if ans is None:
        ans = row.get("quiz_answer")
    ans_s = (ans or "") if ans is not None else ""
    return {
        "exam_quiz_id": row.get("exam_quiz_id"),
        "exam_tab_id": row.get("exam_tab_id") or "",
        "rag_tab_id": row.get("rag_tab_id"),
        "rag_unit_id": row.get("rag_unit_id"),
        "rag_quiz_id": row.get("rag_quiz_id"),
        "person_id": row.get("person_id") or "",
        "course_id": row.get("course_id"),
        "follow_up": bool(row.get("follow_up")),
        "follow_up_exam_quiz_id": row.get("follow_up_exam_quiz_id"),
        "unit_name": row.get("unit_name") or "",
        "quiz_name": row.get("quiz_name") or "",
        "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
        "quiz_content": row.get("quiz_content"),
        "quiz_hint": row.get("quiz_hint"),
        "quiz_answer_reference": row.get("quiz_answer_reference"),
        "quiz_rate": row.get("quiz_rate"),
        "answer_user_prompt_text": row.get("answer_user_prompt_text"),
        "answer_content": ans_s,
        "quiz_answer": ans_s,
        "answer_critique": row.get("answer_critique"),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
    }
