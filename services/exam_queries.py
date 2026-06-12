"""
Exam / Exam_Quiz 資料存取 service。包含 DB 查詢輔助、資料組裝輔助。
routers/exam、routers/course_analysis.py、routers/person_analysis.py 共用，不含任何 FastAPI 路由。
"""

from typing import Any

from postgrest.exceptions import APIError

from utils.taipei_time import now_taipei_iso
from utils.db_schema import (
    ACTIVE_DELETED_FILTER,
    EXAM_COURSE_ID_DEFAULT,
    EXAM_QUIZ_SELECT_COLUMNS,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY_LIST,
    EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST,
    EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY,
    EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST,
    EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST_PROMPT_TEXT,
    RAG_SELECT_COLUMNS,
    RAG_SELECT_COLUMNS_LEGACY,
    RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA,
    RAG_SELECT_COLUMNS_NO_FILE_METADATA,
    apply_parsed_quiz_history_list_tree,
    exam_quiz_list_row,
)
from utils.supabase import get_supabase


# ---------------------------------------------------------------------------
# Rag 表查詢（含 transcript 欄位相容）
# ---------------------------------------------------------------------------

def select_rag_row_with_transcript_fallback(supabase: Any, rag_id: int) -> Any:
    """讀取 public.Rag；SELECT 欄位順序同 public.Rag DDL。缺欄（42703）時依序降級。"""
    candidates = (
        RAG_SELECT_COLUMNS,
        RAG_SELECT_COLUMNS_NO_FILE_METADATA,
        RAG_SELECT_COLUMNS_LEGACY,
        RAG_SELECT_COLUMNS_LEGACY_NO_FILE_METADATA,
        "rag_id, rag_page_id, transcript",
        "rag_id, rag_page_id, transcription",
        "rag_id, rag_page_id",
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
                x in msg
                for x in ("course_id", "file_metadata", "person_id", "tab_name", "transcript", "transcription")
            ):
                continue
            raise
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Exam 資料列建立
# ---------------------------------------------------------------------------

def exam_default_row(
    exam_page_id: str,
    *,
    tab_name: str = "",
    person_id: str = "",
    course_id: int = EXAM_COURSE_ID_DEFAULT,
    local: bool = False,
) -> dict[str, Any]:
    """Exam 表新增一筆時的預設欄位（不含 exam_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    return {
        "exam_page_id": exam_page_id,
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


def exams_by_page_ids(exam_page_ids: list[str]) -> list[dict]:
    """依 exam_page_id 查詢 Exam 表（僅 deleted=False）。"""
    if not exam_page_ids:
        return []
    supabase = get_supabase()
    return (
        supabase.table("Exam")
        .select("*")
        .in_("exam_page_id", exam_page_ids)
        .eq("deleted", False)
        .execute()
        .data or []
    )


# ---------------------------------------------------------------------------
# Exam_Quiz 表查詢
# ---------------------------------------------------------------------------

def apply_exam_quiz_not_deleted(q):
    """僅未軟刪之列（deleted=false 或 null；deleted=true 不回傳，與 Rag_Quiz 相同）。"""
    return q.or_(ACTIVE_DELETED_FILTER)


def _build_exam_quiz_select(
    *,
    columns: str,
    exam_page_ids: list[str] | None = None,
    person_id: str | None = None,
    course_id: int | None = None,
):
    """組 Exam_Quiz select query（不含 deleted 篩選；每次呼叫回傳新 builder）。"""
    supabase = get_supabase()
    q = supabase.table("Exam_Quiz").select(columns)
    if exam_page_ids is not None:
        q = q.in_("exam_page_id", exam_page_ids)
    if person_id is not None:
        q = q.eq("person_id", person_id.strip())
    if course_id is not None:
        q = q.eq("course_id", course_id)
    return q


def _select_exam_quiz_rows(
    *,
    columns: str,
    exam_page_ids: list[str] | None = None,
    person_id: str | None = None,
    course_id: int | None = None,
) -> list[dict]:
    """查 Exam_Quiz（僅未軟刪；deleted=true 不回傳）。"""
    rows = (
        apply_exam_quiz_not_deleted(_build_exam_quiz_select(
            columns=columns,
            exam_page_ids=exam_page_ids,
            person_id=person_id,
            course_id=course_id,
        ))
        .order("exam_quiz_id", desc=False)
        .execute()
        .data
        or []
    )
    return [exam_quiz_list_row(r) for r in rows]


def _strip_answer_rate_from_columns(columns: str) -> str:
    """移除 select 欄位中的 answer_rate（舊版 Exam_Quiz 無此欄時 fallback）。"""
    return ", ".join(
        p.strip() for p in columns.split(",") if p.strip() and p.strip() != "answer_rate"
    )


def _select_exam_quiz_rows_with_follow_up_fallback(**kwargs: Any) -> list[dict]:
    for cols in (
        "*",
        EXAM_QUIZ_SELECT_COLUMNS,
        EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST,
        EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY_LIST_PROMPT_TEXT,
        EXAM_QUIZ_SELECT_COLUMNS_NO_QUIZ_HISTORY,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY_LIST,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_EXAM_QUIZ_ID_NO_QUIZ_HISTORY,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP,
        EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST,
    ):
        try:
            return _select_exam_quiz_rows(columns=cols, **kwargs)
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and "answer_rate" in msg:
                stripped = _strip_answer_rate_from_columns(cols)
                if stripped != cols:
                    try:
                        return _select_exam_quiz_rows(columns=stripped, **kwargs)
                    except APIError as e2:
                        e2_msg = (e2.message or "").lower()
                        if e2.code == "42703" and (
                            "quiz_history_list" in e2_msg
                            or "quiz_history_list_prompt_text" in e2_msg
                            or "follow_up" in e2_msg
                            or "follow_up_exam_quiz_id" in e2_msg
                        ):
                            continue
                        raise
            if e.code == "42703" and (
                "quiz_history_list" in msg
                or "quiz_history_list_prompt_text" in msg
                or "follow_up" in msg
                or "follow_up_exam_quiz_id" in msg
            ):
                continue
            raise
    return _select_exam_quiz_rows(
        columns=EXAM_QUIZ_SELECT_COLUMNS_NO_FOLLOW_UP_NO_QUIZ_HISTORY_LIST, **kwargs
    )


def quizzes_by_exam_page_ids(
    exam_page_ids: list[str],
    *,
    course_id: int | None = None,
) -> dict[str, list[dict]]:
    """依 exam_page_id 查詢 Exam_Quiz，回傳 exam_page_id -> quizzes[]（含 follow_up）。"""
    if not exam_page_ids:
        return {}
    rows = _select_exam_quiz_rows_with_follow_up_fallback(
        exam_page_ids=exam_page_ids, course_id=course_id
    )
    out: dict[str, list[dict]] = {tid: [] for tid in exam_page_ids}
    for row in rows:
        tid = row.get("exam_page_id")
        if tid is not None:
            out.setdefault(str(tid), []).append(row)
    return out


def quizzes_by_person_id(person_id: str, *, course_id: int) -> list[dict]:
    """依 person_id、course_id 查詢 Exam_Quiz（供個人分析使用）。"""
    pid = (person_id or "").strip()
    if not pid:
        return []
    return _select_exam_quiz_rows_with_follow_up_fallback(person_id=pid, course_id=course_id)


def quizzes_by_course_id(course_id: int) -> list[dict]:
    """依 course_id 查詢 Exam_Quiz 全部筆數（供課程分析使用）。"""
    return _select_exam_quiz_rows_with_follow_up_fallback(course_id=course_id)


# ---------------------------------------------------------------------------
# Exam_Quiz 資料組裝輔助
# ---------------------------------------------------------------------------

def _exam_quiz_rag_page_id_str(q: dict) -> str:
    rt = q.get("rag_page_id")
    if rt is None:
        return ""
    return str(rt).strip()


def enrich_exam_quizzes_rag_tab_from_units(quizzes_flat: list[dict]) -> None:
    """Exam_Quiz 列若無 rag_page_id 但有 rag_unit_id，自 Rag_Unit 補上 rag_page_id（就地修改）。"""
    if not quizzes_flat:
        return
    missing_units: set[int] = set()
    for q in quizzes_flat:
        if _exam_quiz_rag_page_id_str(q):
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
        .select("rag_unit_id, rag_page_id")
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
        rtv = row.get("rag_page_id")
        rt = (str(rtv).strip() if rtv is not None else "") or None
        tab_for_unit[uid] = rt
    for q in quizzes_flat:
        if _exam_quiz_rag_page_id_str(q):
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
            q["rag_page_id"] = rt


def ensure_exam_quiz_rag_id_keys(quizzes_flat: list[dict]) -> None:
    """確保每筆 Exam_Quiz dict 皆含 rag_quiz_id、rag_page_id、rag_unit_id 鍵（無則 null）。"""
    for q in quizzes_flat:
        for key in ("rag_quiz_id", "rag_page_id", "rag_unit_id"):
            if key not in q:
                q[key] = None


def nest_follow_up_quizzes(quizzes: list[dict]) -> None:
    """就地為追問鏈附加 follow_up_quiz（由舊題指向新題，exam_quiz_id 遞增）。
    例：Q1.follow_up_quiz → Q2 → Q3（Q1 最舊，在鏈外層／頂層）。"""
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
            prev_q["follow_up_quiz"] = q


def filter_to_chain_roots(quizzes: list[dict]) -> list[dict]:
    """只回傳 chain root 為頂層：follow_up_exam_quiz_id 為 0／空者（含 follow_up=true 但無前一題）。
    其餘 follow_up_exam_quiz_id>0 者嵌於前一題的 follow_up_quiz；前一題已軟刪（不在 quizzes）者一併略過。"""
    out: list[dict] = []
    for q in quizzes:
        try:
            prev_id = int(q.get("follow_up_exam_quiz_id") or 0)
        except (TypeError, ValueError):
            prev_id = 0
        # prev_id>0：此題嵌於前一題的 follow_up_quiz，或前一題已軟刪——兩種情況都不列為頂層。
        if prev_id > 0:
            continue
        out.append(q)
    return out


def chain_root_exam_quiz_id(root: dict) -> int:
    """追問鏈 root（頂層最舊題）的 exam_quiz_id，供排序用。"""
    try:
        return int(root.get("exam_quiz_id") or 0)
    except (TypeError, ValueError):
        return 0


def exam_tab_quizzes_response(quizzes: list[dict]) -> list[dict]:
    """GET /exam/pages 等每筆 Exam 的 quizzes[]：追問 nest、只留 root、exam_quiz_id 升序。"""
    if not quizzes:
        return []
    nest_follow_up_quizzes(quizzes)
    roots = filter_to_chain_roots(quizzes)
    roots.sort(key=chain_root_exam_quiz_id)
    for root in roots:
        apply_parsed_quiz_history_list_tree(root)
    return roots


def exams_with_quizzes_response(quizzes: list[dict]) -> list[dict]:
    """
    依 exam_page_id 將 quizzes 分組組裝為 Exam 列（每筆含 quizzes[]）。
    弱點分析 llm-analysis 回傳的 exams 與 GET /{person,course}-analyses 的 exams 共用此組裝。
    """
    page_ids: list[str] = list(dict.fromkeys(
        str(q.get("exam_page_id")) for q in quizzes if q.get("exam_page_id") is not None
    ))
    exam_rows = exams_by_page_ids(page_ids)
    quizzes_by_tab: dict[str, list[dict]] = {tid: [] for tid in page_ids}
    for q in quizzes:
        tid = q.get("exam_page_id")
        if tid is not None:
            quizzes_by_tab.setdefault(str(tid), []).append(q)

    flat_for_enrich = [qz for tid in page_ids for qz in quizzes_by_tab.get(tid, [])]
    enrich_exam_quizzes_rag_tab_from_units(flat_for_enrich)
    ensure_exam_quiz_rag_id_keys(flat_for_enrich)

    for row in exam_rows:
        tid = str(row.get("exam_page_id") or "")
        row["quizzes"] = exam_tab_quizzes_response(quizzes_by_tab.get(tid, []))
    return exam_rows


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
    rt = row.get("rag_page_id")
    rag_page_id = (str(rt).strip() if rt is not None else "") or None
    return {
        "rag_quiz_id": rag_quiz_id,
        "rag_page_id": rag_page_id,
        "rag_unit_id": rag_unit_id,
        "person_id": str(row.get("person_id") or ""),
        "follow_up": bool(row.get("follow_up")),
        "quiz_name": str(row.get("quiz_name") or ""),
        "quiz_user_prompt_text": str(row.get("quiz_user_prompt_text") or ""),
        "quiz_system_prompt_text": str(row.get("quiz_system_prompt_text") or ""),
        "quiz_content": str(row.get("quiz_content") or ""),
        "quiz_hint": str(row.get("quiz_hint") or ""),
        "quiz_answer_reference": str(row.get("quiz_answer_reference") or ""),
        "answer_user_prompt_text": str(row.get("answer_user_prompt_text") or ""),
    }
