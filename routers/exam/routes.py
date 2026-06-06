"""routers.exam routes（自 exam.py 拆分）。"""

import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse
from postgrest.exceptions import APIError

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.openapi import openapi_body

from services.exam_queries import (
    apply_exam_quiz_not_deleted,
    ensure_exam_quiz_rag_id_keys,
    enrich_exam_quizzes_rag_tab_from_units,
    exam_default_row,
    exams_table_select,
    exam_tab_quizzes_response,
    quizzes_by_exam_page_ids,
    rag_quiz_for_exam_response_row,
)
from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_exam_quiz_with_grade,
)
from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.retry import call_with_transient_http_retry
from utils.serialization import to_json_safe
from utils.llm_key import course_api_key_exists, fetch_api_key_setting_row, get_exam_api_key, get_rag_llm_model
from utils.course_setting import COURSE_SETTING_EXAM_API_KEY
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)
from utils.rag_course import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.rag_exam_setting import is_localhost_request, rag_id_from_rag_page_id, resolve_exam_content_rag_id
from utils.rag_stem import get_rag_stem_from_rag_id, instruction_from_rag_row, transcript_from_row
from utils.supabase import get_supabase
from utils.zip_storage import generate_page_id, get_zip_path
from utils.db_schema import (
    QUIZ_HISTORY_OPENAPI_ITEM,
    QUIZ_HISTORY_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST,
    parse_quiz_history_prompt_text,
    parse_rag_quiz_history_list,
)


from utils.fs import safe_unlink
from .schemas import (
    CreateExamRequest,
    ExamApiKeyExistsResponse,
    ExamApiKeyResponse,
    ExamCreateLlmGenerateQuizFollowupRequest,
    ExamCreateLlmGenerateQuizRequest,
    ExamLlmGenerateQuizFollowupRequest,
    ExamLlmGenerateQuizRequest,
    ExamQuizGradeRateRequest,
    ExamQuizGradeRequest,
    ExamQuizRateRequest,
    ListExamResponse,
    ListRagForExamsResponse,
    PutExamApiKeyRequest,
    UpdateExamUnitNameRequest,
)
from .helpers import (
    _create_exam_quiz_record,
    _exam_llm_generate_quiz_impl,
    _exam_quiz_history_prompt_dicts,
)

_logger = logging.getLogger("routers.exam")


router = APIRouter(prefix="/exam", tags=["exam"])

_exam_grade_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# GET /exam/pages
# ---------------------------------------------------------------------------

@router.get("/pages", response_model=ListExamResponse)
def list_exams(
    request: Request,
    person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 Exam.local 與此值相同的列。未傳時：本機連線視為 true，否則 false",
    ),
):
    """列出 Exam（deleted=false，person_id 篩選，local 篩選）。每筆 Exam 帶 quizzes[]（Exam_Quiz，含 follow_up 鏈、quiz_history_list）。"""

    def _list_exams_once() -> ListExamResponse:
        local_filter = local if local is not None else is_localhost_request(request)
        data = exams_table_select(exclude_deleted=True, local_match=local_filter, course_id=course_id)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        page_ids = list(dict.fromkeys(
            str(r.get("exam_page_id")) for r in data if r.get("exam_page_id") is not None
        ))
        quizzes_by_tab = quizzes_by_exam_page_ids(page_ids, course_id=course_id)
        flat_qz = [qz for tid in page_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_qz)
        ensure_exam_quiz_rag_id_keys(flat_qz)

        for row in data:
            page_id = str(row.get("exam_page_id") or "")
            row["quizzes"] = exam_tab_quizzes_response(quizzes_by_tab.get(page_id, []))

        data = to_json_safe(data)
        return ListExamResponse(exams=data, count=len(data))

    try:
        return call_with_transient_http_retry(_list_exams_once)
    except Exception as e:
        _logger.exception("GET /exam/pages 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Exam 失敗: {e!s}")


# ---------------------------------------------------------------------------
# GET /exam/rag-for-exams
# ---------------------------------------------------------------------------

@router.get(
    "/rag-for-exams",
    response_model=ListRagForExamsResponse,
    summary="List RAG units & quizzes marked for exam",
)
def list_rag_for_exams(
    request: Request,
    _person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 rag_page_id 隸屬 Rag.local 與此值相同之單元／題目。未傳時：本機連線視為 true，否則 false",
    ),
):
    """
    回傳 for-exam 相關 RAG 單元與題目（不限 person_id）：
    - 僅 rag_page_id 對應之 Rag 列（deleted=false）其 local 與 query local 相符者（未傳 local 時同 GET /exam/pages 依連線判定）。
    - 單元：Rag_Unit.deleted=false 且（Rag_Unit.for_exam=true 或至少一筆 Rag_Quiz.for_exam=true 隸屬該 rag_unit_id）。
    - quizzes：僅 Rag_Quiz.for_exam=true 且 deleted=false。
    """

    def _list_rag_for_exams_once() -> ListRagForExamsResponse:
        supabase = get_supabase()
        local_filter = local if local is not None else is_localhost_request(request)

        tabs_for_local = (
            supabase.table("Rag")
            .select("rag_page_id")
            .eq("deleted", False)
            .eq("local", local_filter)
            .eq("course_id", course_id)
            .execute()
            .data
            or []
        )
        allowed_page_ids = list(dict.fromkeys(
            r["rag_page_id"] for r in tabs_for_local if r.get("rag_page_id") is not None
        ))
        if not allowed_page_ids:
            return ListRagForExamsResponse(units=[], count=0)

        def build_exam_quizzes(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, quiz_user_prompt_text, "
                "quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("for_exam", True)
                .eq("deleted", False)
                .in_("rag_page_id", allowed_page_ids)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        quizzes_resp = execute_with_course_id_fallback("Rag_Quiz", build_exam_quizzes, course_id)
        quiz_rows = quizzes_resp.data or []

        unit_ids_from_quizzes: set[int] = set()
        quizzes_by_unit: dict[int, list[dict]] = {}
        for q in quiz_rows:
            q_uid = q.get("rag_unit_id")
            if q_uid is None:
                continue
            try:
                uid = int(q_uid)
            except (TypeError, ValueError):
                continue
            unit_ids_from_quizzes.add(uid)
            quizzes_by_unit.setdefault(uid, []).append(rag_quiz_for_exam_response_row(q))

        unit_ids_from_units: set[int] = set()
        try:
            def build_units_flag(with_course_filter: bool):
                q = (
                    supabase.table("Rag_Unit")
                    .select("rag_unit_id")
                    .eq("for_exam", True)
                    .eq("deleted", False)
                    .in_("rag_page_id", allowed_page_ids)
                )
                if with_course_filter and course_id is not None:
                    q = q.eq("course_id", course_id)
                return q

            units_flag_resp = execute_with_course_id_fallback(
                "Rag_Unit", build_units_flag, course_id
            )
            for u in units_flag_resp.data or []:
                rid = u.get("rag_unit_id")
                if rid is not None:
                    try:
                        unit_ids_from_units.add(int(rid))
                    except (TypeError, ValueError):
                        pass
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and "for_exam" in msg:
                _logger.warning("Rag_Unit 無 for_exam 欄位；GET /exam/rag-for-exams 僅列出含 Rag_Quiz.for_exam 之單元")
            else:
                raise

        all_unit_ids = list(dict.fromkeys(list(unit_ids_from_units | unit_ids_from_quizzes)))
        if not all_unit_ids:
            return ListRagForExamsResponse(units=[], count=0)

        def build_exam_units(with_course_filter: bool):
            q = (
                supabase.table("Rag_Unit")
                .select("*")
                .in_("rag_unit_id", all_unit_ids)
                .in_("rag_page_id", allowed_page_ids)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        units = (
            execute_with_course_id_fallback("Rag_Unit", build_exam_units, course_id).data or []
        )
        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        out = to_json_safe(units)
        return ListRagForExamsResponse(units=out, count=len(out))

    try:
        return call_with_transient_http_retry(_list_rag_for_exams_once)
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("GET /exam/rag-for-exams 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 for_exam RAG 失敗: {e!s}")


# ---------------------------------------------------------------------------
# POST /exam/pages
# ---------------------------------------------------------------------------

@router.post("/pages", status_code=201)
def create_exam(
    body: openapi_body(
        CreateExamRequest,
        {"exam_page_id": "", "person_id": "", "tab_name": "", "local": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """建立一筆 Exam。exam_page_id 可選（未傳由後端產生）；local 選填（預設 false）。"""
    fid = (body.exam_page_id or "").strip()
    body_pid = (body.person_id or "").strip()
    person_id = body_pid if body_pid else caller_person_id
    if body_pid and body_pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與呼叫者（token）不一致")
    if not fid:
        fid = generate_page_id(person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_page_id")
    tab_name = (body.tab_name or "").strip()

    supabase = get_supabase()
    ins = (
        supabase.table("Exam")
        .insert(exam_default_row(fid, tab_name=tab_name, person_id=person_id, course_id=course_id, local=body.local))
        .execute()
    )
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="建立 Exam 失敗")
    row = ins.data[0]
    return {
        "exam_id": row.get("exam_id"),
        "exam_page_id": row.get("exam_page_id", fid),
        "tab_name": row.get("tab_name", tab_name),
        "person_id": row.get("person_id", person_id),
        "course_id": row.get("course_id", course_id),
        "local": row.get("local", body.local),
        "deleted": row.get("deleted", False),
        "updated_at": to_taipei_iso(row.get("updated_at")),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


# ---------------------------------------------------------------------------
# PATCH /exam/pages/{exam_page_id}
# ---------------------------------------------------------------------------

@router.patch("/pages/{exam_page_id}", summary="Update Exam Tab Name")
def update_exam_unit_tab_name(
    body: openapi_body(UpdateExamUnitNameRequest, {"tab_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_page_id: str = PathParam(..., description="要更名的 Exam 的 exam_page_id"),
):
    """更新既有 Exam 的 tab_name（以 exam_page_id 定位；僅 deleted=false）。"""
    fid = (exam_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_page_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Exam")
            .select("exam_id, exam_page_id, tab_name, person_id, course_id, local, deleted")
            .eq("exam_page_id", fid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 exam_page_id 的 Exam 資料，或已刪除")
        row = sel.data[0]
        exam_id = row.get("exam_id")
        pid = row.get("person_id")
        if (pid or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Exam")
        ts = now_taipei_iso()
        supabase.table("Exam").update({"tab_name": tab_name, "updated_at": ts}).eq("exam_page_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
        return {
            "exam_id": exam_id,
            "exam_page_id": fid,
            "tab_name": tab_name,
            "person_id": pid,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /exam/pages/{exam_page_id}
# ---------------------------------------------------------------------------

@router.delete("/pages/{exam_page_id}", status_code=200, summary="Delete Exam Tab", operation_id="exam_tab_delete")
def delete_exam(
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_page_id: str = PathParam(..., description="要刪除的 Exam 的 exam_page_id"),
):
    """DELETE /exam/pages/{exam_page_id}。軟刪除：將 Exam 的 deleted 設為 true。"""
    fid = (exam_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_page_id")
    supabase = get_supabase()
    r = (
        supabase.table("Exam")
        .select("exam_id, person_id")
        .eq("exam_page_id", fid)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="找不到該 exam_page_id 的 Exam 資料，或已刪除")
    pid = (r.data[0].get("person_id") or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權刪除該 Exam")
    supabase.table("Exam").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("exam_page_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
    return {"message": "已將 Exam 標記為刪除", "exam_page_id": fid, "person_id": pid}


# ---------------------------------------------------------------------------
# POST /exam/quizzes/llm-generate
# ---------------------------------------------------------------------------

_EXAM_LLM_GEN_DESCRIPTION = """\
Body：`exam_quiz_id`、`rag_page_id`、`rag_unit_id`、`rag_quiz_id` 皆必填（順序同 public.Exam_Quiz）。
`rag_page_id` 須對應 `public.Rag.rag_page_id`，且與所列 `rag_unit_id`、`rag_quiz_id` 在 DB 上所隸屬之 Tab 一致；並用此載入 ZIP／單元（**不依賴** Course_Setting 之 `rag_localhost`/`rag_deploy`）。
若該 Exam_Quiz 列**已有**有效的 `rag_unit_id`、`rag_quiz_id`，請求兩鍵須與列**完全一致**，否則 400。
若列**尚未**寫入（缺其一或為 0），則以此請求綁定，出題成功後一併寫回。
`quiz_user_prompt_text`／`answer_user_prompt_text` 僅自 Rag_Quiz（請求中的 `rag_quiz_id`）讀取，不另由 body 帶入文字；出題成功後寫入 Exam_Quiz 以記錄當下模板。
unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcript。
unit_type 2／3／4 時不載入 RAG ZIP，改以 transcript 純 LLM 出題。
選填 `quiz_history_list`（八欄位 JSON 物件陣列）：僅寫入 DB；未傳視為空陣列。
選填 `quiz_history_list_prompt_text`（JSON 物件陣列，每筆僅 quiz_content）：併入 LLM 出題 prompt；寫入 DB。
出題成功後更新該筆 Exam_Quiz（`rag_page_id`、`unit_name`（與 RAG 單元顯示名一致，供 GET /exam/pages 分群）、`quiz_name`、quiz_content／quiz_hint／quiz_answer_reference、rag_unit_id、rag_quiz_id；自該 `rag_quiz_id` 之 Rag_Quiz 寫入 `quiz_user_prompt_text`、`answer_user_prompt_text` 以記錄當下模板；清空作答欄位）。

**回應 JSON**（除題目欄位外）必含：`quiz_user_prompt_text`、`answer_user_prompt_text`（與寫入 Exam_Quiz 之快照相同，供前端顯示出題／作答模板）；`unit_name` 與資料庫更新後一致。
"""

_EXAM_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE = list(QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST)
_EXAM_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE = list(QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST)

_EXAM_LLM_GENERATE_OPENAPI_EXAMPLES = {
    "exam_quiz_id": 1,
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _EXAM_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_EXAM_CREATE_LLM_GENERATE_OPENAPI_EXAMPLES = {
    "exam_page_id": "string",
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _EXAM_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_EXAM_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES = {
    "exam_quiz_id": 2,
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "follow_up_exam_quiz_id": 1,
    "quiz_history_list": [
        {**QUIZ_HISTORY_OPENAPI_ITEM, "answer_critique": "批改評語（指出答不好之處）"},
    ],
    "quiz_history_list_prompt_text": _EXAM_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}


_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES = {
    "exam_page_id": "string",
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "follow_up_exam_quiz_id": 1,
    "quiz_history_list": [
        {**QUIZ_HISTORY_OPENAPI_ITEM, "answer_critique": "批改評語（指出答不好之處）"},
    ],
    "quiz_history_list_prompt_text": _EXAM_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}


@router.post(
    "/quizzes/llm-generate",
    summary="Rag LLM Generate Quiz",
    operation_id="exam_llm_generate_quiz",
    description=_EXAM_LLM_GEN_DESCRIPTION.strip(),
)
@router.post("/generate-quiz", include_in_schema=False)
def exam_llm_generate_quiz(
    request: Request,
    body: openapi_body(ExamLlmGenerateQuizRequest, _EXAM_LLM_GENERATE_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """實作與說明見模組常數 `_EXAM_LLM_GEN_DESCRIPTION`（OpenAPI operation description）。亦可改用 POST /exam/quizzes/create-llm-generate 一次完成建立與出題。"""
    _ = request
    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=body.exam_quiz_id,
        rag_page_id=body.rag_page_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_exam_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
    )


# ---------------------------------------------------------------------------
# POST /exam/quizzes/llm-generate-followup
# ---------------------------------------------------------------------------

_EXAM_LLM_GEN_FOLLOWUP_DESCRIPTION = """\
Body：`exam_quiz_id`、`rag_page_id`、`rag_unit_id`、`rag_quiz_id` 必填。
**追問鏈結**：請求 `follow_up_exam_quiz_id`（>0）時，出題成功後寫入本列 `follow_up=true` 與該 id（原樣）。
`quiz_history_list_prompt_text` 非空時才使用追問 LLM prompt；否則仍寫入 follow_up 但出題邏輯同一般 `llm-generate`。
`follow_up_exam_quiz_id` 為 0 或未傳則視為第一題，**回應不含** `follow_up`／`follow_up_exam_quiz_id`。
回應可含 `quiz_history_list`、`quiz_history_list_prompt_text` 與 `created_at`。
其餘 RAG 綁定、unit_type 出題邏輯同 `POST /exam/quizzes/llm-generate`。
`quiz_history_list` 為八欄位 JSON 物件陣列，僅寫入 DB。
`quiz_history_list_prompt_text` 為四欄位 JSON 物件陣列（quiz_content、quiz_answer_reference、answer_content、answer_critique），併入 LLM prompt。
使用 `SYSTEM_PROMPT_QUIZ_FOLLOWUP`／`USER_PROMPT_COURSE_FOLLOWUP`：作答不佳則針對弱點追問，作答良好則改出新的不重複題目。
"""


@router.post(
    "/quizzes/llm-generate-followup",
    summary="Exam LLM Generate Follow-up Quiz",
    operation_id="exam_llm_generate_quiz_followup",
    description=_EXAM_LLM_GEN_FOLLOWUP_DESCRIPTION.strip(),
)
def exam_llm_generate_quiz_followup(
    request: Request,
    body: openapi_body(ExamLlmGenerateQuizFollowupRequest, _EXAM_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """依先前問答接續出下一題；寫入 follow_up 與 follow_up_exam_quiz_id。亦可改用 POST /exam/quizzes/create-llm-generate-followup 一次完成建立與接續出題。"""
    _ = request
    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=body.exam_quiz_id,
        rag_page_id=body.rag_page_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_exam_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
        follow_up_exam_quiz_id=body.follow_up_exam_quiz_id,
    )


# ---------------------------------------------------------------------------
# POST /exam/quizzes/create-llm-generate
# ---------------------------------------------------------------------------

_EXAM_CREATE_LLM_GENERATE_DESCRIPTION = """\
等同先 POST /exam/quizzes 再 POST /exam/quizzes/llm-generate。
Body 不需 `exam_quiz_id`（由 create 產生）；其餘 RAG 綁定、unit_type 出題邏輯與回應 JSON 同 `llm-generate`。
"""


@router.post(
    "/quizzes/create-llm-generate",
    summary="Exam Create Quiz and LLM Generate",
    operation_id="exam_create_llm_generate_quiz",
    description=_EXAM_CREATE_LLM_GENERATE_DESCRIPTION.strip(),
)
def exam_create_llm_generate_quiz(
    request: Request,
    body: openapi_body(ExamCreateLlmGenerateQuizRequest, _EXAM_CREATE_LLM_GENERATE_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """先新增空白 Exam_Quiz，再 LLM 出題；回應同 llm-generate。"""
    _ = request
    try:
        created = _create_exam_quiz_record(
            exam_page_id=body.exam_page_id,
            caller_person_id=caller_person_id,
            course_id=course_id,
        )
        exam_quiz_id = int(created["exam_quiz_id"])
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"建立 Exam_Quiz 後無法取得 exam_quiz_id: {e!s}") from e
    except Exception as e:
        _logger.exception("POST /exam/quizzes/create-llm-generate 建立 Exam_Quiz 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=exam_quiz_id,
        rag_page_id=body.rag_page_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_exam_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
    )


# ---------------------------------------------------------------------------
# POST /exam/quizzes/create-llm-generate-followup
# ---------------------------------------------------------------------------

_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_DESCRIPTION = """\
等同先 POST /exam/quizzes 再 POST /exam/quizzes/llm-generate-followup。
Body 不需 `exam_quiz_id`（由 create 產生）。
出題成功後**一律**寫入本列 `follow_up=true`；`follow_up_exam_quiz_id` 以請求傳入為準（可為 0）。
`quiz_history_list_prompt_text` 非空時使用追問 LLM prompt，否則出題邏輯同一般 llm-generate。
"""


@router.post(
    "/quizzes/create-llm-generate-followup",
    summary="Exam Create Quiz and LLM Generate Follow-up",
    operation_id="exam_create_llm_generate_quiz_followup",
    description=_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_DESCRIPTION.strip(),
)
def exam_create_llm_generate_quiz_followup(
    request: Request,
    body: openapi_body(
        ExamCreateLlmGenerateQuizFollowupRequest,
        _EXAM_CREATE_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES,
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """先新增空白 Exam_Quiz，再依先前問答接續 LLM 出題；回應同 llm-generate-followup。"""
    _ = request
    try:
        created = _create_exam_quiz_record(
            exam_page_id=body.exam_page_id,
            caller_person_id=caller_person_id,
            course_id=course_id,
        )
        exam_quiz_id = int(created["exam_quiz_id"])
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"建立 Exam_Quiz 後無法取得 exam_quiz_id: {e!s}") from e
    except Exception as e:
        _logger.exception("POST /exam/quizzes/create-llm-generate-followup 建立 Exam_Quiz 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=exam_quiz_id,
        rag_page_id=body.rag_page_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_exam_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
        follow_up_exam_quiz_id=body.follow_up_exam_quiz_id,
        always_mark_follow_up=True,
    )


# ---------------------------------------------------------------------------
# POST /exam/quizzes/llm-grade
# ---------------------------------------------------------------------------

@router.post("/quizzes/llm-grade", summary="Exam Grade Quiz", operation_id="exam_llm_grade_quiz")
@router.post("/quizzes/grade", summary="Exam Grade Quiz", include_in_schema=False)
async def exam_grade_submission(
    request: Request,
    background_tasks: BackgroundTasks,
    body: openapi_body(
        ExamQuizGradeRequest,
        {"exam_quiz_id": 1, "quiz_content": "", "quiz_answer": "學生作答文字"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    以 exam_quiz_id 定位題目，進行 RAG+LLM 非同步評分。
    unit_type 2／3／4 時改以 transcript 純 LLM 批改。
    評分 prompt 模板優先用 Exam_Quiz.quiz_user_prompt_text／answer_user_prompt_text（與 POST …/llm-generate 寫入一致），欄位為空時再讀 Rag_Quiz。
    評分完成後直接更新 Exam_Quiz.answer_content / answer_critique。
    回傳 202 + job_id；輪詢 GET /exam/quizzes/grade-result/{job_id}。
    """
    supabase = get_supabase()

    qsel = (
        supabase.table("Exam_Quiz")
        .select(
            "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, unit_name, quiz_name, "
            "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, quiz_rate, "
            "answer_user_prompt_text, answer_content, answer_critique, updated_at, created_at"
        )
        .eq("exam_quiz_id", body.exam_quiz_id)
        .eq("course_id", course_id)
    )
    qsel = apply_exam_quiz_not_deleted(qsel).limit(1).execute()
    if not qsel.data or len(qsel.data) == 0:
        return JSONResponse(
            status_code=404,
            content={"error": f"找不到 exam_quiz_id={body.exam_quiz_id} 的 Exam_Quiz，或已刪除"},
        )
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    rag_unit_id_val = qrow.get("rag_unit_id")
    stored_quiz_content = (qrow.get("quiz_content") or "").strip()

    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Exam_Quiz 評分"})

    quiz_content = (body.quiz_content or "").strip() or stored_quiz_content

    api_key = get_exam_api_key(course_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "請設定 Exam API Key：PUT /exam/llm-api-key（Course_Setting key=exam-api-key，依 course_id）",
            },
        )
    llm_model = get_rag_llm_model(course_id)

    try:
        rag_uid_int = int(rag_unit_id_val) if rag_unit_id_val is not None else 0
    except (TypeError, ValueError):
        rag_uid_int = 0

    rag_rqid_int = 0
    _erq0 = qrow.get("rag_quiz_id")
    if _erq0 is not None:
        try:
            rag_rqid_int = int(_erq0)
            if rag_rqid_int < 0:
                rag_rqid_int = 0
        except (TypeError, ValueError):
            rag_rqid_int = 0

    grade_unit_filter: str | None = (qrow.get("unit_name") or "").strip() or None
    exam_grade_unit_type = 0
    transcript_for_unit = ""
    if rag_uid_int > 0:

        def _grade_unit_sel(cols: str, with_course_filter: bool):
            c = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(c)
                .eq("rag_unit_id", rag_uid_int)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        def _grade_unit_execute(cols: str):
            return execute_with_course_id_fallback(
                "Rag_Unit",
                lambda wc: _grade_unit_sel(cols, wc),
                course_id,
            )

        _gcols_full = (
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _gcols_no_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _gcols_no_fc = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _gcols_min = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _gcols_legacy_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        _gcols_no_fc_legacy_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        try:
            unit_sel = _grade_unit_execute(_gcols_full)
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code != "42703":
                raise
            if "transcript" in msg:
                try:
                    unit_sel = _grade_unit_execute(_gcols_legacy_tr)
                except APIError as e_legacy:
                    if e_legacy.code == "42703" and "transcription" in (e_legacy.message or "").lower():
                        try:
                            unit_sel = _grade_unit_execute(_gcols_no_tr)
                        except APIError as e2:
                            if e2.code == "42703" and "folder_combination" in (e2.message or "").lower():
                                unit_sel = _grade_unit_execute(_gcols_min)
                            else:
                                raise
                    else:
                        raise
            elif "folder_combination" in msg:
                try:
                    unit_sel = _grade_unit_execute(_gcols_no_fc)
                except APIError as e2:
                    if e2.code == "42703" and "transcript" in (e2.message or "").lower():
                        try:
                            unit_sel = _grade_unit_execute(_gcols_no_fc_legacy_tr)
                        except APIError as e3:
                            if e3.code == "42703" and "transcription" in (e3.message or "").lower():
                                unit_sel = _grade_unit_execute(_gcols_min)
                            else:
                                raise
                    else:
                        raise
            else:
                raise
        if unit_sel.data:
            u0 = unit_sel.data[0]
            path_key = (u0.get("folder_combination") or u0.get("unit_name") or "").strip()
            if path_key:
                grade_unit_filter = path_key
            try:
                exam_grade_unit_type = int(u0.get("unit_type") or 0)
            except (TypeError, ValueError):
                exam_grade_unit_type = 0
            transcript_for_unit = transcript_from_row(u0)

    rag_id_used: int | None = None
    rt_exam = (str(qrow.get("rag_page_id") or "").strip())
    if rt_exam:
        rag_id_used = rag_id_from_rag_page_id(supabase, rt_exam, course_id)

    if rag_id_used is None or rag_id_used <= 0:
        rag_id_used, _ = resolve_exam_content_rag_id(
            supabase,
            request,
            stem_rag_unit_id=rag_uid_int if rag_uid_int > 0 else None,
            rag_quiz_id=rag_rqid_int if rag_rqid_int > 0 else None,
            course_id=course_id,
        )

    if rag_id_used is None or rag_id_used <= 0:
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    "無法決定評分用之 RAG：請確認 Exam_Quiz 填有對應 `Rag` 之 rag_page_id，"
                    "或 rag_unit_id／rag_quiz_id 可自 Rag_Unit／Rag_Quiz 解析；仍可於 Course_Setting "
                    "設定 rag_localhost／rag_deploy，value=Rag.rag_id。"
                ),
            },
        )

    try:
        rag_id = int(rag_id_used)
        row_exam, _stem, rag_zip_page_id = get_rag_stem_from_rag_id(
            supabase,
            rag_id,
            include_row=True,
            unit_name=grade_unit_filter,
            rag_unit_id=rag_uid_int if rag_uid_int > 0 else None,
        )
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    assert_row_course_id(row_exam, course_id, "Rag")

    transcript_text = (transcript_for_unit or instruction_from_rag_row(row_exam)).strip()

    quiz_user_prompt_exam = (qrow.get("quiz_user_prompt_text") or "").strip()
    answer_user_prompt_exam = (qrow.get("answer_user_prompt_text") or "").strip()
    exam_rag_quiz_id: int | None = rag_rqid_int if rag_rqid_int > 0 else None
    try:
        need_quiz = not quiz_user_prompt_exam
        need_answer = not answer_user_prompt_exam
        if (need_quiz or need_answer) and rag_rqid_int > 0:
            def build_rqgx(with_course_filter: bool):
                cols = select_without_course_id_if_needed(
                    "Rag_Quiz",
                    "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, quiz_user_prompt_text, "
                    "quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
                    with_course_filter,
                )
                q = (
                    supabase.table("Rag_Quiz")
                    .select(cols)
                    .eq("rag_quiz_id", rag_rqid_int)
                    .eq("deleted", False)
                )
                if with_course_filter and course_id is not None:
                    q = q.eq("course_id", course_id)
                return q.limit(1)

            rqgx = execute_with_course_id_fallback("Rag_Quiz", build_rqgx, course_id)
            if rqgx.data:
                r0 = rqgx.data[0]
                if need_quiz:
                    quiz_user_prompt_exam = (r0.get("quiz_user_prompt_text") or "").strip()
                if need_answer:
                    answer_user_prompt_exam = (r0.get("answer_user_prompt_text") or "").strip()
    except (TypeError, ValueError):
        pass

    transcript_grade: str | None = None

    if exam_grade_unit_type in (2, 3, 4):
        if not transcript_text:
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcript 未設定（單元 2／3／4）；請於 Rag_Unit 或 Rag 設定 transcript"},
            )
        transcript_grade = transcript_text
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_exam_grade_tx_"))
    else:
        rag_zip_path = get_zip_path(rag_zip_page_id)
        if not rag_zip_path or not rag_zip_path.exists():
            return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP（page_id={rag_zip_page_id}）"})
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_exam_grade_"))
        zip_source_path = work_dir / "ref.zip"
        extract_folder = work_dir / "extract"
        extract_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(rag_zip_path, zip_source_path)
            if not zipfile.is_zipfile(zip_source_path):
                cleanup_grade_workspace(work_dir)
                return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
        except Exception as e:
            cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            safe_unlink(rag_zip_path)

    job_id = str(uuid.uuid4())
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None, "llm_error": None}
    exam_quiz_id_int = int(body.exam_quiz_id)
    def insert_fn(rd, qa):
        return update_exam_quiz_with_grade(
            rd, qa, exam_quiz_id=exam_quiz_id_int, grade_llm_model=llm_model
        )
    background_tasks.add_task(
        run_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        body.quiz_answer or "",
        _exam_grade_job_results,
        insert_fn,
        answer_user_prompt_exam,
        exam_quiz_id=exam_quiz_id_int,
        rag_quiz_id=exam_rag_quiz_id,
        unit_type=exam_grade_unit_type,
        transcript_grade=transcript_grade,
        quiz_user_prompt_text=quiz_user_prompt_exam,
        llm_model=llm_model,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "grade_llm_model": llm_model})


# ---------------------------------------------------------------------------
# GET /exam/quizzes/grade-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/quizzes/grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str, _person_id: PersonId, course_id: CourseId):
    """
    輪詢 Exam 評分結果（搭配 POST /exam/quizzes/llm-grade）。
    status: pending | ready | error；ready 時 result 含 quiz_comments、exam_quiz_id。
    """
    if job_id not in _exam_grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _exam_grade_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "llm_error": data.get("llm_error"),
    }
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            eid = res.get("exam_quiz_id")
            if eid is not None:
                try:
                    eid_int = int(eid)
                    if eid_int > 0:
                        supabase = get_supabase()
                        q = apply_exam_quiz_not_deleted(
                            supabase.table("Exam_Quiz")
                            .select("*")
                            .eq("exam_quiz_id", eid_int)
                            .eq("course_id", course_id)
                        ).limit(1).execute()
                        if q.data:
                            exam_quiz_row = to_json_safe(q.data[0])
                            if isinstance(exam_quiz_row, dict):
                                exam_quiz_row["quiz_history_list"] = parse_rag_quiz_history_list(
                                    exam_quiz_row.get("quiz_history_list")
                                )
                                exam_quiz_row["quiz_history_list_prompt_text"] = (
                                    parse_quiz_history_prompt_text(
                                        exam_quiz_row.get("quiz_history_list_prompt_text"),
                                        followup=bool(exam_quiz_row.get("follow_up")),
                                    )
                                )
                            out["exam_quiz"] = exam_quiz_row
                except (TypeError, ValueError):
                    pass
    return out


# ---------------------------------------------------------------------------
# PUT /exam/quizzes/{exam_quiz_id}/quiz-rate
# ---------------------------------------------------------------------------

@router.put("/quizzes/{exam_quiz_id}/quiz-rate", summary="Exam Rate Quiz", status_code=200)
def update_exam_quiz_rate(
    body: openapi_body(ExamQuizRateRequest, {"quiz_rate": 0}),
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_quiz_id: int = PathParam(..., ge=1, description="Exam_Quiz 主鍵"),
):
    """依 exam_quiz_id 更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。"""
    quiz_rate = int(body.quiz_rate)
    supabase = get_supabase()
    r = apply_exam_quiz_not_deleted(
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, person_id, course_id")
        .eq("exam_quiz_id", exam_quiz_id)
        .eq("course_id", course_id)
    ).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={exam_quiz_id} 的 Exam_Quiz，或已刪除")
    qpid = (r.data[0].get("person_id") or "").strip()
    if qpid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權更新該題 quiz_rate")
    supabase.table("Exam_Quiz").update(
        {"quiz_rate": quiz_rate, "updated_at": now_taipei_iso()}
    ).eq("exam_quiz_id", exam_quiz_id).execute()
    after = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, quiz_rate, updated_at, created_at")
        .eq("exam_quiz_id", exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not after.data or len(after.data) == 0:
        raise HTTPException(status_code=500, detail="更新 quiz_rate 後讀取失敗")
    out = dict(after.data[0])
    out["message"] = "已更新 quiz_rate"
    return to_json_safe(out)


# ---------------------------------------------------------------------------
# PUT /exam/quizzes/{exam_quiz_id}/grade-rate
# ---------------------------------------------------------------------------

@router.put("/quizzes/{exam_quiz_id}/grade-rate", summary="Exam Rate Grade", status_code=200)
def update_exam_quiz_grade_rate(
    body: openapi_body(ExamQuizGradeRateRequest, {"grade_rate": 0}),
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_quiz_id: int = PathParam(..., ge=1, description="Exam_Quiz 主鍵"),
):
    """依 exam_quiz_id 更新 Exam_Quiz.grade_rate（僅 -1、0、1）。"""
    grade_rate = int(body.grade_rate)
    supabase = get_supabase()
    r = apply_exam_quiz_not_deleted(
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, person_id, course_id")
        .eq("exam_quiz_id", exam_quiz_id)
        .eq("course_id", course_id)
    ).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={exam_quiz_id} 的 Exam_Quiz，或已刪除")
    qpid = (r.data[0].get("person_id") or "").strip()
    if qpid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權更新該題 grade_rate")
    supabase.table("Exam_Quiz").update(
        {"grade_rate": grade_rate, "updated_at": now_taipei_iso()}
    ).eq("exam_quiz_id", exam_quiz_id).execute()
    after = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, grade_rate, updated_at, created_at")
        .eq("exam_quiz_id", exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not after.data or len(after.data) == 0:
        raise HTTPException(status_code=500, detail="更新 grade_rate 後讀取失敗")
    out = dict(after.data[0])
    out["message"] = "已更新 grade_rate"
    return to_json_safe(out)


# ---------------------------------------------------------------------------
# DELETE /exam/quizzes/{exam_quiz_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/quizzes/{exam_quiz_id}",
    status_code=200,
    summary="Delete Exam Quiz",
    operation_id="exam_tab_quiz_delete",
)
def delete_exam_quiz(
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_quiz_id: int = PathParam(..., gt=0, description="要軟刪除的 Exam_Quiz 主鍵"),
):
    """
    DELETE /exam/quizzes/{exam_quiz_id}。
    軟刪除：將 Exam_Quiz 該列 deleted 設為 true（僅 person_id 與請求者一致且尚未刪除之列）。
    """
    try:
        supabase = get_supabase()

        def build_quiz_delete_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Exam_Quiz",
                "exam_quiz_id, exam_page_id, person_id, course_id",
                with_course_filter,
            )
            q = apply_exam_quiz_not_deleted(
                supabase.table("Exam_Quiz")
                .select(cols)
                .eq("exam_quiz_id", exam_quiz_id)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Exam_Quiz", build_quiz_delete_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 exam_quiz_id 的 Exam_Quiz 資料，或已刪除")
        row = sel.data[0]
        pid = (row.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Exam_Quiz")
        ts = now_taipei_iso()
        ids_to_delete: set[int] = {exam_quiz_id}
        while True:
            child_resp = apply_exam_quiz_not_deleted(
                supabase.table("Exam_Quiz")
                .select("exam_quiz_id")
                .in_("follow_up_exam_quiz_id", list(ids_to_delete))
            ).execute()
            new_ids: set[int] = set()
            for child_row in child_resp.data or []:
                cid = child_row.get("exam_quiz_id")
                if cid is None:
                    continue
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                if cid_int not in ids_to_delete:
                    new_ids.add(cid_int)
            if not new_ids:
                break
            ids_to_delete |= new_ids
        apply_exam_quiz_not_deleted(
            supabase.table("Exam_Quiz")
            .update({"deleted": True, "updated_at": ts})
            .in_("exam_quiz_id", list(ids_to_delete))
        ).execute()
        return {
            "message": "已將 Exam_Quiz 標記為刪除",
            "exam_quiz_id": exam_quiz_id,
            "exam_page_id": row.get("exam_page_id"),
            "person_id": pid,
            "exam_quiz_updated": True,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /exam/quizzes/{exam_quiz_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/llm-api-key/exists", response_model=ExamApiKeyExistsResponse)
def get_exam_api_key_exists(person_id: PersonId, course_id: CourseId):
    """查詢 Exam LLM API Key 是否已設定（Course_Setting key=exam-api-key，依 course_id）；不回傳 key 內容。"""
    _require_active_person(person_id)
    return ExamApiKeyExistsResponse(
        course_id=course_id,
        exists=course_api_key_exists(COURSE_SETTING_EXAM_API_KEY, course_id),
    )


@router.get("/llm-api-key", response_model=ExamApiKeyResponse)
def get_exam_api_key_setting(person_id: PersonId, course_id: CourseId):
    """讀取 Exam LLM API Key（Course_Setting key=exam-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_api_key_setting_row(COURSE_SETTING_EXAM_API_KEY, course_id)
    if not row:
        return ExamApiKeyResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return ExamApiKeyResponse(
        course_setting_id=row.get("course_setting_id"),
        course_id=course_id,
        api_key=value or None,
    )


@router.put("/llm-api-key", response_model=ExamApiKeyResponse)
def put_exam_api_key_setting(
    body: openapi_body(PutExamApiKeyRequest, {"api_key": "sk-..."}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 Exam LLM API Key（Course_Setting key=exam-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.api_key or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            COURSE_SETTING_EXAM_API_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            return ExamApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return ExamApiKeyResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=course_id,
            api_key=saved or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
