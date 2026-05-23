"""
Exam / Exam_Quiz 資料存取 service。包含 DB 查詢輔助、資料組裝輔助。
routers/exam.py、routers/course_analysis.py、routers/person_analysis.py 共用，不含任何 FastAPI 路由。
"""

import json
from typing import Any

from postgrest.exceptions import APIError

from utils.datetime_utils import now_taipei_iso
from utils.db_tables import (
    EXAM_COURSE_ID_DEFAULT,
    EXAM_QUIZ_SELECT_COLUMNS,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID,
    RAG_SELECT_COLUMNS,
    RAG_SELECT_COLUMNS_LEGACY,
    RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA,
    RAG_SELECT_COLUMNS_NO_FILE_METADATA,
    exam_quiz_list_row,
)
from utils.supabase_client import get_supabase


# ---------------------------------------------------------------------------
# Rag 表查詢（含 transcription 欄位相容）
# ---------------------------------------------------------------------------

def select_rag_row_with_transcription_fallback(supabase: Any, rag_id: int) -> Any:
    """讀取 public.Rag；SELECT 欄位順序同 public.Rag DDL。缺欄（42703）時依序降級。"""
    candidates = (
        RAG_SELECT_COLUMNS,
        RAG_SELECT_COLUMNS_NO_FILE_METADATA,
        RAG_SELECT_COLUMNS_LEGACY,
        RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA,
        "rag_id, rag_tab_id, transcription",
        "rag_id, rag_tab_id",
    )
    last_err: APIError | None = None
    for cols in candidates:
        try:
            return (
                supabase.table("Rag")
                .select(cols)
                .eq("rag_id", rag_id)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
        except APIError as e:
            last_err = e
            msg = (e.message or "").lower()
            if e.code == "42703" and any(
                x in msg for x in ("course_id", "file_metadata", "person_id", "tab_name", "transcription")
            ):
                continue
            raise
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Exam 資料列建立
# ---------------------------------------------------------------------------

def exam_default_row(
    exam_tab_id: str,
    *,
    tab_name: str = "",
    person_id: str = "",
    course_id: int = EXAM_COURSE_ID_DEFAULT,
    local: bool = False,
) -> dict[str, Any]:
    """Exam 表新增一筆時的預設欄位（不含 exam_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    return {
        "exam_tab_id": exam_tab_id,
        "tab_name": tab_name,
        "person_id": person_id,
        "course_id": course_id,
        "local": local,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }


# ---------------------------------------------------------------------------
# Exam 表查詢
# ---------------------------------------------------------------------------

def exams_table_select(
    exclude_deleted: bool = True,
    *,
    local_match: bool | None = None,
    course_id: int | None = None,
) -> list[dict]:
    """查詢 Exam 表。exclude_deleted=True 時僅回傳 deleted=False；course_id 若指定則僅回傳該課程。依 created_at 升序。"""
    supabase = get_supabase()
    q = supabase.table("Exam").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    if course_id is not None:
        q = q.eq("course_id", course_id)
    q = q.order("created_at", desc=False)
    return q.execute().data or []


def exams_by_ids(exam_ids: list[int]) -> list[dict]:
    """依 exam_id 查詢 Exam 表（僅 deleted=False）。"""
    if not exam_ids:
        return []
    supabase = get_supabase()
    return (
        supabase.table("Exam")
        .select("*")
        .in_("exam_id", exam_ids)
        .eq("deleted", False)
        .execute()
        .data or []
    )


def exams_by_tab_ids(exam_tab_ids: list[str]) -> list[dict]:
    """依 exam_tab_id 查詢 Exam 表（僅 deleted=False）。"""
    if not exam_tab_ids:
        return []
    supabase = get_supabase()
    return (
        supabase.table("Exam")
        .select("*")
        .in_("exam_tab_id", exam_tab_ids)
        .eq("deleted", False)
        .execute()
        .data or []
    )


# ---------------------------------------------------------------------------
# Exam_Quiz 表查詢
# ---------------------------------------------------------------------------

def _select_exam_quiz_rows(
    *,
    columns: str,
    exam_tab_ids: list[str] | None = None,
    person_id: str | None = None,
    course_id: int | None = None,
) -> list[dict]:
    """查 Exam_Quiz；回傳列經 exam_quiz_list_row 正規化（含 follow_up）。"""
    supabase = get_supabase()
    q = supabase.table("Exam_Quiz").select(columns)
    if exam_tab_ids is not None:
        q = q.in_("exam_tab_id", exam_tab_ids)
    if person_id is not None:
        q = q.eq("person_id", person_id.strip())
    if course_id is not None:
        q = q.eq("course_id", course_id)
    rows = q.order("created_at", desc=False).execute().data or []
    return [exam_quiz_list_row(r) for r in rows]


def _select_exam_quiz_rows_with_follow_up_fallback(**kwargs: Any) -> list[dict]:
    for cols in (
        EXAM_QUIZ_SELECT_COLUMNS,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP,
    ):
        try:
            return _select_exam_quiz_rows(columns=cols, **kwargs)
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and ("follow_up" in msg or "follow_up_exam_quiz_id" in msg):
                continue
            raise
    return _select_exam_quiz_rows(columns=EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP, **kwargs)


def quizzes_by_exam_tab_ids(
    exam_tab_ids: list[str],
    *,
    course_id: int | None = None,
) -> dict[str, list[dict]]:
    """依 exam_tab_id 查詢 Exam_Quiz，回傳 exam_tab_id -> quizzes[]（含 follow_up）。"""
    if not exam_tab_ids:
        return {}
    rows = _select_exam_quiz_rows_with_follow_up_fallback(
        exam_tab_ids=exam_tab_ids, course_id=course_id
    )
    out: dict[str, list[dict]] = {tid: [] for tid in exam_tab_ids}
    for row in rows:
        tid = row.get("exam_tab_id")
        if tid is not None:
            out.setdefault(str(tid), []).append(row)
    return out


def quizzes_by_person_id(person_id: str) -> list[dict]:
    """依 person_id 查詢 Exam_Quiz 全部筆數（供分析使用）。"""
    pid = (person_id or "").strip()
    if not pid:
        return []
    return _select_exam_quiz_rows_with_follow_up_fallback(person_id=pid)


def all_exam_quizzes() -> list[dict]:
    """查詢 Exam_Quiz 表全部筆數（供課程分析使用）。"""
    return _select_exam_quiz_rows_with_follow_up_fallback()


# ---------------------------------------------------------------------------
# Exam_Quiz 資料組裝輔助
# ---------------------------------------------------------------------------

def _exam_quiz_rag_tab_id_str(q: dict) -> str:
    rt = q.get("rag_tab_id")
    if rt is None:
        return ""
    return str(rt).strip()


def enrich_exam_quizzes_rag_tab_from_units(quizzes_flat: list[dict]) -> None:
    """Exam_Quiz 列若無 rag_tab_id 但有 rag_unit_id，自 Rag_Unit 補上 rag_tab_id（就地修改）。"""
    if not quizzes_flat:
        return
    missing_units: set[int] = set()
    for q in quizzes_flat:
        if _exam_quiz_rag_tab_id_str(q):
            continue
        ru = q.get("rag_unit_id")
        if ru is None:
            continue
        try:
            missing_units.add(int(ru))
        except (TypeError, ValueError):
            pass
    if not missing_units:
        return
    supabase = get_supabase()
    resp = (
        supabase.table("Rag_Unit")
        .select("rag_unit_id, rag_tab_id")
        .in_("rag_unit_id", list(missing_units))
        .eq("deleted", False)
        .execute()
    )
    tab_for_unit: dict[int, str | None] = {}
    for row in resp.data or []:
        rid = row.get("rag_unit_id")
        if rid is None:
            continue
        try:
            uid = int(rid)
        except (TypeError, ValueError):
            continue
        rtv = row.get("rag_tab_id")
        rt = (str(rtv).strip() if rtv is not None else "") or None
        tab_for_unit[uid] = rt
    for q in quizzes_flat:
        if _exam_quiz_rag_tab_id_str(q):
            continue
        ru = q.get("rag_unit_id")
        if ru is None:
            continue
        try:
            uid = int(ru)
        except (TypeError, ValueError):
            continue
        rt = tab_for_unit.get(uid)
        if rt:
            q["rag_tab_id"] = rt


def ensure_exam_quiz_rag_id_keys(quizzes_flat: list[dict]) -> None:
    """確保每筆 Exam_Quiz dict 皆含 rag_quiz_id、rag_tab_id、rag_unit_id 鍵（無則 null）。"""
    for q in quizzes_flat:
        for key in ("rag_quiz_id", "rag_tab_id", "rag_unit_id"):
            if key not in q:
                q[key] = None


def nest_follow_up_quizzes(quizzes: list[dict]) -> None:
    """就地為 follow_up=True 的 quiz 附加 follow_up_quiz（指向前一題 dict，遞迴串接）。
    quizzes 須依 created_at 升序，確保處理到 C 時 B 已附加 follow_up_quiz=A。"""
    by_id: dict[int, dict] = {}
    for q in quizzes:
        qid = q.get("exam_quiz_id")
        if qid is not None:
            try:
                by_id[int(qid)] = q
            except (TypeError, ValueError):
                pass
    for q in quizzes:
        if not q.get("follow_up"):
            continue
        prev_id_raw = q.get("follow_up_exam_quiz_id")
        if not prev_id_raw:
            continue
        try:
            prev_id = int(prev_id_raw)
        except (TypeError, ValueError):
            continue
        if prev_id <= 0:
            continue
        prev_q = by_id.get(prev_id)
        if prev_q is not None:
            q["follow_up_quiz"] = prev_q


def filter_to_chain_heads(quizzes: list[dict]) -> list[dict]:
    """只回傳 chain head：被其他 quiz 的 follow_up_exam_quiz_id 引用的 quiz 不出現頂層（已含於 follow_up_quiz 鏈中）。"""
    referenced_ids: set[int] = set()
    for q in quizzes:
        prev_id_raw = q.get("follow_up_exam_quiz_id")
        if not prev_id_raw:
            continue
        try:
            prev_id = int(prev_id_raw)
            if prev_id > 0:
                referenced_ids.add(prev_id)
        except (TypeError, ValueError):
            pass
    return [q for q in quizzes if int(q.get("exam_quiz_id") or 0) not in referenced_ids]


def group_exam_quizzes_into_units(quizzes: list[dict]) -> list[dict]:
    """依 Exam_Quiz.unit_name 分群；units[] 每筆含 unit_name、rag_unit_id（取自首筆有效值）、quizzes[]。"""
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for q in quizzes:
        un = (q.get("unit_name") or "").strip()
        if un not in buckets:
            order.append(un)
            buckets[un] = []
        buckets[un].append(q)
    units_out: list[dict] = []
    for un in order:
        qlist = buckets[un]
        rag_uid: int | None = None
        for q in qlist:
            ru = q.get("rag_unit_id")
            if ru is None:
                continue
            try:
                if int(ru) > 0:
                    rag_uid = int(ru)
                    break
            except (TypeError, ValueError):
                pass
        units_out.append({"unit_name": un, "rag_unit_id": rag_uid, "quizzes": qlist})
    return units_out


def rag_quiz_for_exam_response_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rag_Quiz 列：for-exam 端點回傳 RAG 關聯鍵與出題／批改 prompt。"""
    ru = row.get("rag_unit_id")
    try:
        rag_unit_id: int | None = int(ru) if ru is not None else None
    except (TypeError, ValueError):
        rag_unit_id = None
    rq = row.get("rag_quiz_id")
    try:
        rag_quiz_id: int | None = int(rq) if rq is not None else None
    except (TypeError, ValueError):
        rag_quiz_id = None
    rt = row.get("rag_tab_id")
    rag_tab_id = (str(rt).strip() if rt is not None else "") or None
    return {
        "rag_quiz_id": rag_quiz_id,
        "rag_tab_id": rag_tab_id,
        "rag_unit_id": rag_unit_id,
        "person_id": str(row.get("person_id") or ""),
        "follow_up": bool(row.get("follow_up")),
        "quiz_name": str(row.get("quiz_name") or ""),
        "quiz_user_prompt_text": str(row.get("quiz_user_prompt_text") or ""),
        "quiz_content": str(row.get("quiz_content") or ""),
        "quiz_hint": str(row.get("quiz_hint") or ""),
        "quiz_answer_reference": str(row.get("quiz_answer_reference") or ""),
        "answer_user_prompt_text": str(row.get("answer_user_prompt_text") or ""),
    }
