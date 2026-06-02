"""
與 Supabase／Postgres 實際 schema 一致的表名常數。
若遷移或重命名表，僅需改此處。
"""

from __future__ import annotations

import json
from typing import Any

# public."User"
USER_TABLE = "User"
COLLEGE_TABLE = "College"
COURSE_TABLE = "Course"
# enrollments / per-course profile（user_type）
USER_COURSE_RELATION_TABLE = "User_Course_Relation"

# Supabase PostgREST：deleted 為 false 或 null 視為有效列（與舊資料相容）
ACTIVE_DELETED_FILTER = "deleted.eq.false,deleted.is.null"

# RAG 相關表名
RAG_TABLE = "Rag"
RAG_UNIT_TABLE = "Rag_Unit"
RAG_QUIZ_TABLE = "Rag_Quiz"

# 三表共用 course_id（bigint null default 0）
RAG_COURSE_ID_DEFAULT = 0

# SELECT 欄位順序同 public DDL（rag_page_id → person_id → course_id → …）
RAG_SELECT_COLUMNS = (
    "rag_id, rag_page_id, person_id, course_id, tab_name, file_size, file_metadata, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_NO_FILE_METADATA = (
    "rag_id, rag_page_id, person_id, course_id, tab_name, file_size, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_LEGACY = (
    "rag_id, rag_page_id, person_id, tab_name, file_size, file_metadata, "
    "local, deleted, updated_at, created_at"
)
RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA = (
    "rag_id, rag_page_id, person_id, tab_name, file_size, "
    "local, deleted, updated_at, created_at"
)

RAG_UNIT_SELECT_COLUMNS = (
    "rag_unit_id, rag_page_id, person_id, course_id, unit_name, folder_combination, unit_type, "
    "repack_file_name, rag_file_name, rag_file_size, rag_chunk_size, rag_chunk_overlap, "
    "transcript, text_file_name, mp3_file_name, youtube_url, deleted, updated_at, created_at"
)

RAG_QUIZ_SELECT_COLUMNS = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, quiz_history_list, "
    "for_exam, deleted, updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, quiz_history_list, "
    "for_exam, deleted, updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, for_exam, deleted, "
    "updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, for_exam, deleted, "
    "updated_at, created_at"
)


def _normalize_rag_quiz_history_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    qc = (item.get("quiz_content") or "").strip()
    if not qc:
        return None
    ans = item.get("answer_content")
    if ans is None:
        ans = item.get("quiz_answer") or item.get("answer")
    return {
        "quiz_content": qc,
        "answer_content": (ans or "") if ans is not None else "",
        "quiz_answer_reference": (item.get("quiz_answer_reference") or "").strip(),
        "answer_critique": (item.get("answer_critique") or "").strip(),
    }


def parse_rag_quiz_history_list(raw: Any) -> list[dict[str, str]]:
    """解析 Rag_Quiz／Exam_Quiz 之 quiz_history_list（text 存 JSON 陣列）。"""
    if raw is None or raw == "":
        return []
    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for item in data:
        normalized = _normalize_rag_quiz_history_item(item)
        if normalized is not None:
            out.append(normalized)
    return out


def serialize_rag_quiz_history_list(items: list[dict[str, str]] | None) -> str | None:
    """寫入 Rag_Quiz／Exam_Quiz 之 quiz_history_list；空陣列回傳 null。"""
    if not items:
        return None
    cleaned = [_normalize_rag_quiz_history_item(item) for item in items]
    cleaned = [item for item in cleaned if item is not None]
    if not cleaned:
        return None
    return json.dumps(cleaned, ensure_ascii=False)


def rag_quiz_history_stems(history: list[dict[str, str]] | None) -> list[str]:
    """自 quiz_history_list 物件陣列取出已出過題幹（供一般出題 prompt）。"""
    return [
        (item.get("quiz_content") or "").strip()
        for item in (history or [])
        if (item.get("quiz_content") or "").strip()
    ]


def quiz_history_from_stems(stems: list[str] | None) -> list[dict[str, str]]:
    """將一般出題請求的字串題幹陣列轉為 DB quiz_history_list 物件格式。"""
    out: list[dict[str, str]] = []
    for stem in stems or []:
        s = (stem or "").strip()
        if not s:
            continue
        out.append(
            {
                "quiz_content": s,
                "answer_content": "",
                "quiz_answer_reference": "",
                "answer_critique": "",
            }
        )
    return out


def resolve_quiz_history_for_generate(
    *,
    request_qa: list[dict[str, str]] | None = None,
    request_stems: list[str] | None = None,
    db_raw: Any = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    合併請求 body 與 DB 的 quiz_history_list（請求 QA 物件 > 請求題幹字串 > DB）。
    回傳 (物件陣列，供寫入 DB／追問 LLM；題幹字串陣列，供一般出題 LLM)。
    """
    db_history = parse_rag_quiz_history_list(db_raw)
    qa: list[dict[str, str]] = list(request_qa or [])
    if not qa and request_stems:
        qa = quiz_history_from_stems(request_stems)
    if not qa and db_history:
        qa = list(db_history)
    stems = rag_quiz_history_stems(qa) if qa else [s.strip() for s in (request_stems or []) if (s or "").strip()]
    if not stems and db_history:
        stems = rag_quiz_history_stems(db_history)
    return qa, stems


def apply_parsed_quiz_history_list(quiz: dict[str, Any]) -> None:
    """就地將 quiz 列的 quiz_history_list 轉為物件陣列（GET /tabs 等回傳用）。"""
    quiz["quiz_history_list"] = parse_rag_quiz_history_list(quiz.get("quiz_history_list"))


def apply_parsed_quiz_history_list_tree(quiz: dict[str, Any]) -> None:
    """遞迴處理 Exam follow_up_quiz 鏈上的 quiz_history_list。"""
    apply_parsed_quiz_history_list(quiz)
    child = quiz.get("follow_up_quiz")
    if isinstance(child, dict):
        apply_parsed_quiz_history_list_tree(child)


def rag_quiz_list_row(row: dict[str, Any]) -> dict[str, Any]:
    """GET /rag/tabs、GET /rag/tab/units 之 quizzes[] 單筆（欄位順序對齊 public.Rag_Quiz）。"""
    ans = row.get("answer_content")
    if ans is None:
        ans = row.get("quiz_answer")
    ans_s = (ans or "") if ans is not None else ""
    return {
        "rag_quiz_id": row.get("rag_quiz_id"),
        "rag_page_id": row.get("rag_page_id") or "",
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
        "quiz_history_list": parse_rag_quiz_history_list(row.get("quiz_history_list")),
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
    "exam_id, exam_page_id, person_id, course_id, tab_name, local, deleted, updated_at, created_at"
)

EXAM_QUIZ_SELECT_COLUMNS = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, quiz_history_list, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, "
    "quiz_history_list, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, "
    "updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "quiz_rate, answer_user_prompt_text, answer_content, answer_critique, quiz_history_list, "
    "updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
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
        "exam_page_id": row.get("exam_page_id") or "",
        "rag_page_id": row.get("rag_page_id"),
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
        "quiz_history_list": parse_rag_quiz_history_list(row.get("quiz_history_list")),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
    }
