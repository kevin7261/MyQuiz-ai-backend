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

# Bank 相關表名（自 Rag/Rag_Unit 複製之獨立題庫；程式不與 rag 共用）
BANK_TABLE = "Bank"
BANK_UNIT_TABLE = "Bank_Unit"

# Bank／Bank_Unit 共用 course_id（bigint null default 0）
BANK_COURSE_ID_DEFAULT = 0

BANK_SELECT_COLUMNS = (
    "bank_id, bank_page_id, person_id, course_id, tab_name, file_size, file_metadata, "
    "deleted, updated_at, created_at"
)

BANK_UNIT_SELECT_COLUMNS = (
    "bank_unit_id, bank_page_id, person_id, course_id, unit_name, folder_combination, unit_type, "
    "repack_file_name, upload_file_name, upload_file_size, rag_chunk_size, rag_chunk_overlap, "
    "transcript, text_file_name, mp3_file_name, youtube_url, deleted, updated_at, created_at"
)

# Bank_Group（測試題組）／Bank_QA（測試問答）：對應 rag 的 Rag_Quiz LLM 出題／批改，但以「題組」為單位、無追問
BANK_GROUP_TABLE = "Bank_Group"
BANK_QA_TABLE = "Bank_QA"

BANK_GROUP_SELECT_COLUMNS = (
    "bank_group_id, bank_page_id, bank_unit_id, person_id, course_id, group_name, "
    "question_system_prompt_text, question_user_prompt_text, qa_count, question_llm_model, "
    "answer_user_prompt_text, answer_llm_model, for_exam, deleted, updated_at, created_at"
)

BANK_QA_SELECT_COLUMNS = (
    "bank_qa_id, bank_page_id, bank_unit_id, bank_group_id, person_id, course_id, "
    "question_series_index, question_system_prompt_text, question_user_prompt_text, "
    "question_content, question_hint, question_answer_reference, question_reason, question_llm_model, "
    "answer_user_prompt_text, answer_llm_model, answer_content, answer_critique, "
    "deleted, updated_at, created_at"
)

# Quiz（試卷／Test）：搭配 bank 出題的應試層，定位等同 exam 之於 rag。
# Quiz（試卷）→ Quiz_Group（自 Bank_Group 快照之題組）→ Quiz_QA（逐題出題／批改，無追問）。
QUIZ_TABLE = "Quiz"
QUIZ_GROUP_TABLE = "Quiz_Group"
QUIZ_QA_TABLE = "Quiz_QA"

QUIZ_SELECT_COLUMNS = (
    "quiz_id, quiz_page_id, person_id, course_id, tab_name, deleted, updated_at, created_at"
)

QUIZ_GROUP_SELECT_COLUMNS = (
    "quiz_group_id, quiz_page_id, bank_page_id, bank_unit_id, bank_group_id, person_id, course_id, "
    "unit_name, unit_type, group_name, question_system_prompt_text, question_user_prompt_text, "
    "qa_count, question_llm_model, answer_user_prompt_text, answer_llm_model, "
    "deleted, updated_at, created_at"
)

QUIZ_QA_SELECT_COLUMNS = (
    "quiz_qa_id, quiz_group_id, quiz_page_id, bank_page_id, bank_unit_id, bank_group_id, "
    "person_id, course_id, question_series_index, question_system_prompt_text, question_user_prompt_text, "
    "question_content, question_hint, question_answer_reference, question_reason, question_rate, "
    "question_llm_model, answer_user_prompt_text, answer_llm_model, answer_content, answer_critique, "
    "answer_rate, deleted, updated_at, created_at"
)

# Quiz_Ask（追問）：出題後對題組對應之 Bank 課程內容發問，每問一列
QUIZ_ASK_TABLE = "Quiz_Ask"

QUIZ_ASK_SELECT_COLUMNS = (
    "quiz_ask_id, quiz_group_id, quiz_page_id, bank_page_id, bank_unit_id, bank_group_id, "
    "person_id, course_id, unit_name, unit_type, group_name, "
    "ask_user_prompt_text, answer_content, answer_rate, deleted, updated_at, created_at"
)

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
    "answer_user_prompt_text, answer_content, answer_critique, quiz_llm_model, answer_llm_model, "
    "quiz_history_list, quiz_history_list_prompt_text, for_exam, deleted, updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, quiz_llm_model, answer_llm_model, "
    "quiz_history_list, quiz_history_list_prompt_text, for_exam, deleted, updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, quiz_llm_model, answer_llm_model, "
    "for_exam, deleted, updated_at, created_at"
)
RAG_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST = (
    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, "
    "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "answer_user_prompt_text, answer_content, answer_critique, quiz_llm_model, answer_llm_model, "
    "for_exam, deleted, updated_at, created_at"
)


def _normalize_rag_quiz_history_item(item: Any) -> dict[str, Any] | None:
    """正規化單筆 quiz_history_list 為八欄位物件（quiz_content 必填）。"""
    if isinstance(item, str):
        qc = item.strip()
        if not qc:
            return None
        return quiz_history_item(quiz_content=qc)
    if not isinstance(item, dict):
        return None
    qc = (item.get("quiz_content") or "").strip()
    if not qc:
        return None
    ans = item.get("answer_content")
    if ans is None:
        ans = item.get("quiz_answer") or item.get("answer")
    rag_unit_id = item.get("rag_unit_id")
    try:
        rag_unit_id = int(rag_unit_id or 0)
    except (TypeError, ValueError):
        rag_unit_id = 0
    follow_up_raw = item.get("follow_up")
    if follow_up_raw is None:
        follow_up_raw = item.get("followup") or item.get("followUp")
    return quiz_history_item(
        rag_unit_id=rag_unit_id,
        quiz_name=(item.get("quiz_name") or "").strip(),
        follow_up=bool(follow_up_raw),
        quiz_content=qc,
        quiz_hint=(item.get("quiz_hint") or item.get("hint") or "").strip(),
        quiz_answer_reference=(item.get("quiz_answer_reference") or "").strip(),
        answer_content=(ans or "") if ans is not None else "",
        answer_critique=(item.get("answer_critique") or "").strip(),
    )


def quiz_history_item(
    *,
    rag_unit_id: int = 0,
    quiz_name: str = "",
    follow_up: bool = False,
    quiz_content: str,
    quiz_hint: str = "",
    quiz_answer_reference: str = "",
    answer_content: str = "",
    answer_critique: str = "",
) -> dict[str, Any]:
    """建立標準 quiz_history_list 單筆（八鍵皆存在）。"""
    return {
        "rag_unit_id": int(rag_unit_id or 0),
        "quiz_name": (quiz_name or "").strip(),
        "follow_up": bool(follow_up),
        "quiz_content": (quiz_content or "").strip(),
        "quiz_hint": (quiz_hint or "").strip(),
        "quiz_answer_reference": (quiz_answer_reference or "").strip(),
        "answer_content": (answer_content or "") if answer_content is not None else "",
        "answer_critique": (answer_critique or "").strip(),
    }


QUIZ_HISTORY_OPENAPI_ITEM: dict[str, Any] = {
    "rag_unit_id": 1,
    "quiz_name": "題型名稱",
    "follow_up": False,
    "quiz_content": "先前題目題幹",
    "quiz_hint": "提示",
    "quiz_answer_reference": "參考答案全文",
    "answer_content": "學生先前作答",
    "answer_critique": "批改評語",
}

QUIZ_HISTORY_OPENAPI_LIST: list[dict[str, Any]] = [dict(QUIZ_HISTORY_OPENAPI_ITEM)]

QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM: dict[str, str] = {
    "quiz_content": "先前題目題幹",
}
QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM: dict[str, str] = {
    "quiz_content": "先前題目題幹",
    "quiz_answer_reference": "參考答案全文",
    "answer_content": "學生先前作答",
    "answer_critique": "批改評語",
}
QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST: list[dict[str, str]] = [
    dict(QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM)
]
QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST: list[dict[str, str]] = [
    dict(QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM)
]


def _normalize_prompt_stem_item(item: Any) -> dict[str, str] | None:
    """正規化 quiz_history_list_prompt_text 單筆（一般出題：僅 quiz_content）。"""
    if isinstance(item, str):
        qc = item.strip()
        if not qc:
            return None
        return {"quiz_content": qc}
    if not isinstance(item, dict):
        return None
    qc = (item.get("quiz_content") or "").strip()
    if not qc:
        return None
    return {"quiz_content": qc}


def _normalize_prompt_followup_item(item: Any) -> dict[str, str] | None:
    """正規化 quiz_history_list_prompt_text 單筆（追問：四欄位）。"""
    if isinstance(item, str):
        qc = item.strip()
        if not qc:
            return None
        return {
            "quiz_content": qc,
            "quiz_answer_reference": "",
            "answer_content": "",
            "answer_critique": "",
        }
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
        "quiz_answer_reference": (item.get("quiz_answer_reference") or "").strip(),
        "answer_content": (ans or "") if ans is not None else "",
        "answer_critique": (item.get("answer_critique") or "").strip(),
    }


def parse_quiz_history_prompt_text(raw: Any, *, followup: bool) -> list[dict[str, str]]:
    """解析 Rag_Quiz.quiz_history_list_prompt_text（text JSON 或已解析 list）。"""
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
    normalize = _normalize_prompt_followup_item if followup else _normalize_prompt_stem_item
    out: list[dict[str, str]] = []
    for item in data:
        normalized = normalize(item)
        if normalized is not None:
            out.append(normalized)
    return out


def serialize_quiz_history_prompt_text(
    items: list[dict[str, Any]] | None,
    *,
    followup: bool,
) -> str:
    """寫入 Rag_Quiz.quiz_history_list_prompt_text（text 欄位用 JSON 字串）。"""
    if not items:
        return "[]"
    normalize = _normalize_prompt_followup_item if followup else _normalize_prompt_stem_item
    cleaned = [normalize(item) for item in items]
    cleaned = [item for item in cleaned if item is not None]
    if not cleaned:
        return "[]"
    return json.dumps(cleaned, ensure_ascii=False)


def coerce_quiz_history_prompt_text_request(raw: Any, *, followup: bool) -> list[dict[str, str]]:
    """請求 body 之 quiz_history_list_prompt_text：接受 JSON 字串或物件陣列。"""
    return parse_quiz_history_prompt_text(raw, followup=followup)


def parse_rag_quiz_history_list(raw: Any) -> list[dict[str, Any]]:
    """解析 Rag_Quiz／Exam_Quiz 之 quiz_history_list（text JSON 或已解析 list；元素可為字串或物件）。"""
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
    out: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_rag_quiz_history_item(item)
        if normalized is not None:
            out.append(normalized)
    return out


def serialize_rag_quiz_history_list(items: list[dict[str, Any]] | None) -> str:
    """寫入 Rag_Quiz／Exam_Quiz 之 quiz_history_list（text 欄位用 JSON 字串）。"""
    if not items:
        return "[]"
    cleaned = [_normalize_rag_quiz_history_item(item) for item in items]
    cleaned = [item for item in cleaned if item is not None]
    if not cleaned:
        return "[]"
    return json.dumps(cleaned, ensure_ascii=False)


def rag_quiz_history_stems(history: list[dict[str, Any]] | None) -> list[str]:
    """自 quiz_history_list 物件陣列取出已出過題幹（供一般出題 prompt）。"""
    return [
        (item.get("quiz_content") or "").strip()
        for item in (history or [])
        if (item.get("quiz_content") or "").strip()
    ]


def quiz_history_from_stems(stems: list[str] | None) -> list[dict[str, Any]]:
    """相容舊 API 字串題幹陣列 → 標準八欄位物件陣列。"""
    out: list[dict[str, Any]] = []
    for stem in stems or []:
        item = _normalize_rag_quiz_history_item(stem)
        if item is not None:
            out.append(item)
    return out


def coerce_quiz_history_request(raw: Any) -> list[dict[str, Any]]:
    """請求 body 之 quiz_history_list：接受 JSON 字串、物件陣列或字串陣列。"""
    if raw is None:
        return []
    return parse_rag_quiz_history_list(raw)


def resolve_quiz_history_for_generate(
    *,
    request_history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    出題僅使用 API 請求 body 的 quiz_history_list（不讀 DB、不自行合成）。
    回傳 (標準物件陣列供寫 DB／追問 LLM；題幹字串陣列供一般出題 LLM)。
    """
    qa = list(request_history or [])
    stems = rag_quiz_history_stems(qa)
    return qa, stems


def apply_parsed_quiz_history_list(quiz: dict[str, Any]) -> None:
    """就地將 quiz 列的 quiz_history_list／quiz_history_list_prompt_text 轉為物件陣列（GET /tabs 等回傳用）。"""
    quiz["quiz_history_list"] = parse_rag_quiz_history_list(quiz.get("quiz_history_list"))
    if "quiz_history_list_prompt_text" in quiz:
        quiz["quiz_history_list_prompt_text"] = parse_quiz_history_prompt_text(
            quiz.get("quiz_history_list_prompt_text"),
            followup=bool(quiz.get("follow_up")),
        )


def apply_parsed_quiz_history_list_tree(quiz: dict[str, Any]) -> None:
    """遞迴處理 Exam follow_up_quiz 鏈上的 quiz_history_list。"""
    apply_parsed_quiz_history_list(quiz)
    child = quiz.get("follow_up_quiz")
    if isinstance(child, dict):
        apply_parsed_quiz_history_list_tree(child)


def rag_quiz_list_row(row: dict[str, Any]) -> dict[str, Any]:
    """GET /v1/rag/pages、GET /v1/rag/pages/{rag_page_id}/units 之 quizzes[] 單筆（欄位順序對齊 public.Rag_Quiz）。"""
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
        "quiz_llm_model": row.get("quiz_llm_model"),
        "answer_llm_model": row.get("answer_llm_model"),
        "quiz_history_list": parse_rag_quiz_history_list(row.get("quiz_history_list")),
        "quiz_history_list_prompt_text": parse_quiz_history_prompt_text(
            row.get("quiz_history_list_prompt_text"),
            followup=bool(row.get("follow_up")),
        ),
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
    "answer_critique, answer_rate, quiz_llm_model, answer_llm_model, quiz_history_list, "
    "quiz_history_list_prompt_text, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, answer_rate, quiz_llm_model, answer_llm_model, quiz_history_list_prompt_text, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST_PROMPT_TEXT = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, answer_rate, quiz_llm_model, answer_llm_model, quiz_history_list, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, follow_up_exam_quiz_id, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, "
    "quiz_hint, quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, "
    "answer_critique, answer_rate, quiz_llm_model, answer_llm_model, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, answer_rate, "
    "quiz_llm_model, answer_llm_model, quiz_history_list, quiz_history_list_prompt_text, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, answer_rate, "
    "quiz_llm_model, answer_llm_model, quiz_history_list_prompt_text, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "follow_up, unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, "
    "quiz_answer_reference, quiz_rate, answer_user_prompt_text, answer_content, answer_critique, answer_rate, "
    "quiz_llm_model, answer_llm_model, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "quiz_rate, answer_user_prompt_text, answer_content, answer_critique, answer_rate, quiz_llm_model, answer_llm_model, "
    "quiz_history_list, quiz_history_list_prompt_text, updated_at, created_at"
)
EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST = (
    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
    "unit_name, quiz_name, quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, "
    "quiz_rate, answer_user_prompt_text, answer_content, answer_critique, answer_rate, quiz_llm_model, answer_llm_model, "
    "updated_at, created_at"
)


def exam_quiz_list_row(row: dict[str, Any]) -> dict[str, Any]:
    """GET /exam/pages 等之 quizzes[] 單筆（欄位順序對齊 public.Exam_Quiz）。"""
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
        "answer_rate": 0 if row.get("answer_rate") is None else row.get("answer_rate"),
        "quiz_llm_model": row.get("quiz_llm_model"),
        "answer_llm_model": row.get("answer_llm_model"),
        "quiz_history_list": parse_rag_quiz_history_list(row.get("quiz_history_list")),
        "quiz_history_list_prompt_text": parse_quiz_history_prompt_text(
            row.get("quiz_history_list_prompt_text"),
            followup=bool(row.get("follow_up")),
        ),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
    }
