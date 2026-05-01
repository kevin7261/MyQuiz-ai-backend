"""
Exam / Exam_Quiz 資料存取 service。包含 DB 查詢輔助、資料組裝輔助。
routers/exam.py、routers/course_analysis.py、routers/person_analysis.py 共用，不含任何 FastAPI 路由。
"""

import json
from typing import Any

from postgrest.exceptions import APIError

from utils.datetime_utils import now_taipei_iso
from utils.supabase_client import get_supabase


# ---------------------------------------------------------------------------
# Rag 表查詢（含 transcription 欄位相容）
# ---------------------------------------------------------------------------

def select_rag_row_with_transcription_fallback(supabase: Any, rag_id: int) -> Any:
    """Rag 表部分環境尚無 transcription 欄位（42703）時改選 rag_id, rag_tab_id。"""
    try:
        return (
            supabase.table("Rag")
            .select("rag_id, rag_tab_id, transcription")
            .eq("rag_id", rag_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "transcription" in msg:
            return (
                supabase.table("Rag")
                .select("rag_id, rag_tab_id")
                .eq("rag_id", rag_id)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
        raise


# ---------------------------------------------------------------------------
# Exam 資料列建立
# ---------------------------------------------------------------------------

def exam_default_row(
    exam_tab_id: str,
    *,
    tab_name: str = "",
    person_id: str = "",
    local: bool = False,
) -> dict[str, Any]:
    """Exam 表新增一筆時的預設欄位（不含 exam_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    return {
        "exam_tab_id": exam_tab_id,
        "tab_name": tab_name,
        "person_id": person_id,
        "local": local,
        "deleted": False,
        "created_at": ts,
        "updated_at": ts,
    }


# ---------------------------------------------------------------------------
# Exam 表查詢
# ---------------------------------------------------------------------------

def exams_table_select(
    exclude_deleted: bool = True,
    *,
    local_match: bool | None = None,
) -> list[dict]:
    """查詢 Exam 表。exclude_deleted=True 時僅回傳 deleted=False；依 created_at 升序。"""
    supabase = get_supabase()
    q = supabase.table("Exam").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
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

def quizzes_by_exam_tab_ids(exam_tab_ids: list[str]) -> dict[str, list[dict]]:
    """依 exam_tab_id 查詢 Exam_Quiz，回傳 exam_tab_id -> list of quiz（依 created_at 升序）。"""
    if not exam_tab_ids:
        return {}
    supabase = get_supabase()
    rows = (
        supabase.table("Exam_Quiz")
        .select("*")
        .in_("exam_tab_id", exam_tab_ids)
        .order("created_at", desc=False)
        .execute()
        .data or []
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
    supabase = get_supabase()
    return supabase.table("Exam_Quiz").select("*").eq("person_id", pid).execute().data or []


def all_exam_quizzes() -> list[dict]:
    """查詢 Exam_Quiz 表全部筆數（供課程分析使用）。"""
    supabase = get_supabase()
    return supabase.table("Exam_Quiz").select("*").execute().data or []


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
        "quiz_name": str(row.get("quiz_name") or ""),
        "quiz_user_prompt_text": str(row.get("quiz_user_prompt_text") or ""),
        "answer_user_prompt_text": str(row.get("answer_user_prompt_text") or ""),
    }
