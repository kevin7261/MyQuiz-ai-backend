"""routers.answer routes（自 answer.py 拆分）。"""

import logging
from typing import Any

from utils.openapi import openapi_body

from fastapi import APIRouter, BackgroundTasks, HTTPException
from dependencies.person_id import PersonId
from dependencies.course_id import CourseId
from fastapi.responses import JSONResponse

from utils.taipei_time import now_taipei_iso
from utils.serialization import to_json_safe
from utils.llm_key import course_api_key_exists, fetch_api_key_setting_row
from utils.course_setting import COURSE_SETTING_RAG_API_KEY, COURSE_SETTING_LLM_MODEL
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)
from utils.rag_course import (
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.supabase import get_supabase
from utils.db_schema import (
    QUIZ_HISTORY_OPENAPI_ITEM,
    QUIZ_HISTORY_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST,
    parse_rag_quiz_history_list,
)


from .schemas import (
    GenerateQuizDbOnlyRequest,
    GenerateQuizFollowupDbOnlyRequest,
    GenerateQuizFollowupRequest,
    GenerateQuizRequest,
    PutRagApiKeyRequest,
    PutRagLlmModelRequest,
    QuizAnswerDbOnlyRequest,
    QuizAnswerRequest,
    RagApiKeyExistsResponse,
    RagApiKeyResponse,
    RagLlmModelResponse,
    RagQuizFollowupRequest,
    RagQuizForExamRequest,
)
from .helpers import (
    _enqueue_rag_llm_answer_job,
    _answer_job_results,
    _quiz_history_prompt_dicts,
    _rag_llm_generate_quiz_impl,
)

_logger = logging.getLogger("routers.answer")


router = APIRouter(prefix="/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# PUT /rag/quizzes/{rag_quiz_id}/followup
# ---------------------------------------------------------------------------


@router.put("/quizzes/{rag_quiz_id}/followup", summary="Set Rag Quiz follow_up flag", operation_id="rag_quiz_followup")
def mark_rag_quiz_followup(
    rag_quiz_id: int,
    body: openapi_body(
        RagQuizFollowupRequest,
        {"followup": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.follow_up（followup=true 標記追問、false 取消）。以 rag_quiz_id（path）定位；僅 deleted=false 且 person_id 一致者可更新。"""
    try:
        supabase = get_supabase()

        def build_followup_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_followup_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz，或已刪除")

        row0 = sel.data[0]
        pid = (row0.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權更新該 Rag_Quiz")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update(
            {"follow_up": body.followup, "updated_at": ts}
        ).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        row = (read.data or [{}])[0]
        return to_json_safe({
            "rag_quiz_id": row.get("rag_quiz_id"),
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": row.get("person_id"),
            "quiz_name": row.get("quiz_name"),
            "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
            "quiz_content": row.get("quiz_content"),
            "quiz_hint": row.get("quiz_hint"),
            "quiz_answer_reference": row.get("quiz_answer_reference"),
            "answer_user_prompt_text": row.get("answer_user_prompt_text"),
            "answer_content": row.get("answer_content"),
            "quiz_answer": row.get("answer_content") or row.get("quiz_answer"),
            "answer_critique": row.get("answer_critique"),
            "for_exam": row.get("for_exam"),
            "follow_up": row.get("follow_up"),
            "deleted": row.get("deleted"),
            "updated_at": row.get("updated_at"),
            "created_at": row.get("created_at"),
        })
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /rag/quizzes/{rag_quiz_id}/followup 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /rag/quizzes/llm-generate
# ---------------------------------------------------------------------------

_RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE = list(QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST)
_RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE = list(QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST)

_RAG_LLM_GENERATE_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": [
        {**QUIZ_HISTORY_OPENAPI_ITEM, "answer_critique": "批改評語（指出答不好之處）"},
    ],
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}

_RAG_LLM_GENERATE_DB_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_RAG_LLM_GENERATE_FOLLOWUP_DB_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_history_list": list(_RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE["quiz_history_list"]),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}


@router.post("/quizzes/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def rag_llm_generate_quiz(
    body: openapi_body(GenerateQuizRequest, _RAG_LLM_GENERATE_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    Body：rag_quiz_id、quiz_name、quiz_user_prompt_text（可空字串）、quiz_history_list（選填；對齊 public.Rag_Quiz 欄位）；
    rag_page_id／rag_unit_id 由後端依 rag_quiz_id 自資料庫帶入；quiz_user_prompt_text 空則自該列 Rag_Quiz 讀取。
    選填 `quiz_history_list`（八欄位 JSON 物件陣列）：僅寫入 DB。
    選填 `quiz_history_list_prompt_text`（JSON 物件陣列，每筆僅 quiz_content）：併入 LLM 出題 prompt；寫入 DB。
    unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcript。
    unit_type 2／3／4 時不載入 RAG ZIP，改以逐字稿為 context；與 unit_type=1 共用 `SYSTEM_PROMPT_QUIZ`、`USER_PROMPT_COURSE` 與 `_generate_quiz_from_context`。
    出題成功後更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=false、quiz_history_list；清空 answer_content、answer_critique；保留 answer_user_prompt_text）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
    )


# ---------------------------------------------------------------------------
# POST /rag/quizzes/llm-generate-followup
# ---------------------------------------------------------------------------


@router.post(
    "/quizzes/llm-generate-followup",
    summary="Rag LLM Generate Follow-up Quiz",
    operation_id="rag_llm_generate_quiz_followup",
)
def rag_llm_generate_quiz_followup(
    body: openapi_body(GenerateQuizFollowupRequest, _RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    依先前問答接續出下一題：作答不佳則針對弱點追問；作答良好則改出新的不重複題目。
    Body 與 `llm-generate` 類似：`quiz_history_list` 僅寫 DB；`quiz_history_list_prompt_text`（四欄位 JSON 物件陣列）併入 LLM prompt。
    `quiz_history_list` 為八欄位物件陣列（見 OpenAPI Example）。
    使用 `SYSTEM_PROMPT_QUIZ_FOLLOWUP`／`USER_PROMPT_COURSE_FOLLOWUP`。
    出題成功後同樣更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=true；寫入 quiz_history_list 為請求或 DB 既有之先前問答；清空 answer_content、answer_critique；保留 answer_user_prompt_text）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
    )


# ---------------------------------------------------------------------------
# POST /rag/quizzes/llm-generate-db
# ---------------------------------------------------------------------------


@router.post(
    "/quizzes/llm-generate-db",
    summary="Rag LLM Generate Quiz (stored quiz_user_prompt_text)",
    operation_id="rag_llm_generate_quiz_db_prompt",
)
def rag_llm_generate_quiz_db_prompt(
    body: openapi_body(GenerateQuizDbOnlyRequest, _RAG_LLM_GENERATE_DB_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`（行為等同傳空字串至 `llm-generate`）。
    出題成功後清空 answer_content、answer_critique；保留 answer_user_prompt_text。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
    )


# ---------------------------------------------------------------------------
# POST /rag/quizzes/llm-generate-followup-db
# ---------------------------------------------------------------------------


@router.post(
    "/quizzes/llm-generate-followup-db",
    summary="Rag LLM Generate Follow-up Quiz (stored quiz_user_prompt_text)",
    operation_id="rag_llm_generate_quiz_followup_db_prompt",
)
def rag_llm_generate_quiz_followup_db_prompt(
    body: openapi_body(GenerateQuizFollowupDbOnlyRequest, _RAG_LLM_GENERATE_FOLLOWUP_DB_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate-followup` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`。
    出題成功後清空 answer_content、answer_critique；保留 answer_user_prompt_text。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
    )


@router.post("/quizzes/llm-answer", summary="Rag Answer Quiz")
async def answer_submission(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizAnswerRequest,
        {
            "rag_id": "1",
            "rag_page_id": "",
            "rag_quiz_id": "1",
            "quiz_content": "",
            "answer_user_prompt_text": "",
            "quiz_answer": "學生作答文字",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    非同步評分：Body 以 rag_id、rag_quiz_id 為核心；quiz_content 可省略（自 Rag_Quiz 讀）。
    `answer_user_prompt_text` 以請求為準（可空；空字串會寫入並覆蓋 Rag_Quiz 該列）。
    unit_type 2／3／4 時以 transcript 純 LLM 批改；其餘依 rag_id 載入 RAG ZIP。
    回傳 202 + job_id；輪詢 GET /rag/quizzes/answer-result/{job_id}。
    """
    return await _enqueue_rag_llm_answer_job(
        background_tasks,
        caller_person_id,
        course_id,
        rag_id_str=body.rag_id,
        rag_quiz_id_str=body.rag_quiz_id,
        qc_from_body=body.quiz_content,
        quiz_answer=body.quiz_answer,
        answer_user_prompt_mode="from_request",
        answer_user_prompt_from_request=(body.answer_user_prompt_text or "").strip(),
    )


@router.post(
    "/quizzes/llm-answer-db",
    summary="Rag Answer Quiz (stored answer_user_prompt_text)",
    operation_id="rag_llm_answer_quiz_db_prompt",
)
async def answer_submission_stored_answer_prompt(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizAnswerDbOnlyRequest,
        {
            "rag_id": "1",
            "rag_page_id": "",
            "rag_quiz_id": "1",
            "quiz_content": "",
            "quiz_answer": "學生作答文字",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-answer` 相同，但請求不含 `answer_user_prompt_text`；
    評分時與寫回皆以 Rag_Quiz 該列既有之 `answer_user_prompt_text` 為準。
    """
    return await _enqueue_rag_llm_answer_job(
        background_tasks,
        caller_person_id,
        course_id,
        rag_id_str=body.rag_id,
        rag_quiz_id_str=body.rag_quiz_id,
        qc_from_body=body.quiz_content,
        quiz_answer=body.quiz_answer,
        answer_user_prompt_mode="from_rag_quiz_row",
    )


# ---------------------------------------------------------------------------
# PUT /rag/quizzes/{rag_quiz_id}/for-exam
# ---------------------------------------------------------------------------

@router.put("/quizzes/{rag_quiz_id}/for-exam", summary="Set Rag Quiz for_exam flag")
def mark_rag_quiz_for_exam(
    rag_quiz_id: int,
    body: openapi_body(
        RagQuizForExamRequest,
        {"for_exam": True},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.for_exam（true＝測驗用、false＝取消）。以 rag_quiz_id（path）定位；僅 deleted=false 且 person_id 一致者可更新。"""
    try:
        supabase = get_supabase()
        def build_for_exam_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_for_exam_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz，或已刪除")

        row0 = sel.data[0]
        pid = (row0.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權更新該 Rag_Quiz")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"for_exam": body.for_exam, "updated_at": ts}).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        row = (read.data or [{}])[0]
        return to_json_safe({
            "rag_quiz_id": row.get("rag_quiz_id"),
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": row.get("person_id"),
            "quiz_name": row.get("quiz_name"),
            "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
            "quiz_content": row.get("quiz_content"),
            "quiz_hint": row.get("quiz_hint"),
            "quiz_answer_reference": row.get("quiz_answer_reference"),
            "answer_user_prompt_text": row.get("answer_user_prompt_text"),
            "answer_content": row.get("answer_content"),
            "quiz_answer": row.get("answer_content") or row.get("quiz_answer"),
            "answer_critique": row.get("answer_critique"),
            "for_exam": row.get("for_exam"),
            "follow_up": row.get("follow_up"),
            "deleted": row.get("deleted"),
            "updated_at": row.get("updated_at"),
            "created_at": row.get("created_at"),
        })
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /rag/quizzes/{rag_quiz_id}/for-exam 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /rag/quizzes/answer-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/quizzes/answer-result/{job_id}", summary="Get Answer Result", tags=["rag"])
async def get_answer_result(job_id: str, _person_id: PersonId, course_id: CourseId):
    """
    輪詢評分結果。status: pending | ready | error；
    ready 時 result 為 quiz_comments、rag_quiz_id（另含 rag_answer_id），並自資料庫讀取 rag_quiz 整列。
    """
    if job_id not in _answer_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _answer_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "llm_error": data.get("llm_error"),
    }
    rag_quiz_row: dict[str, Any] | None = None
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            rid = res.get("rag_quiz_id") or res.get("rag_answer_id")
            if rid is not None:
                try:
                    rid_int = int(rid)
                    if rid_int > 0:
                        supabase = get_supabase()

                        def build_answer_result_sel(with_course_filter: bool):
                            q = (
                                supabase.table("Rag_Quiz")
                                .select("*")
                                .eq("rag_quiz_id", rid_int)
                                .eq("deleted", False)
                            )
                            if with_course_filter and course_id is not None:
                                q = q.eq("course_id", course_id)
                            return q.limit(1)

                        q = execute_with_course_id_fallback(
                            "Rag_Quiz", build_answer_result_sel, course_id
                        )
                        if q.data:
                            rag_quiz_row = to_json_safe(q.data[0])
                            if isinstance(rag_quiz_row, dict):
                                rag_quiz_row["quiz_history_list"] = parse_rag_quiz_history_list(
                                    rag_quiz_row.get("quiz_history_list")
                                )
                except (TypeError, ValueError) as e:
                    _logger.debug("answer-result rag_quiz_id 無效 job_id=%s: %s", job_id, e)
                except Exception as e:
                    _logger.warning("answer-result 讀取 Rag_Quiz 失敗 job_id=%s: %s", job_id, e)
        out["rag_quiz"] = rag_quiz_row
    return out


@router.get("/llm-api-key/exists", response_model=RagApiKeyExistsResponse)
def get_rag_api_key_exists(person_id: PersonId, course_id: CourseId):
    """查詢 RAG LLM API Key 是否已設定（Course_Setting key=rag-api-key，依 course_id）；不回傳 key 內容。"""
    _require_active_person(person_id)
    return RagApiKeyExistsResponse(
        course_id=course_id,
        exists=course_api_key_exists(COURSE_SETTING_RAG_API_KEY, course_id),
    )


@router.get("/llm-api-key", response_model=RagApiKeyResponse)
def get_rag_api_key_setting(person_id: PersonId, course_id: CourseId):
    """讀取 RAG LLM API Key（Course_Setting key=rag-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_api_key_setting_row(COURSE_SETTING_RAG_API_KEY, course_id)
    if not row:
        return RagApiKeyResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return RagApiKeyResponse(
        course_setting_id=row.get("course_setting_id"),
        course_id=course_id,
        api_key=value or None,
    )


@router.put("/llm-api-key", response_model=RagApiKeyResponse)
def put_rag_api_key_setting(
    body: openapi_body(PutRagApiKeyRequest, {"api_key": "sk-..."}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 RAG LLM API Key（Course_Setting key=rag-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.api_key or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            COURSE_SETTING_RAG_API_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            return RagApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return RagApiKeyResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=course_id,
            api_key=saved or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/llm-model", response_model=RagLlmModelResponse)
def get_rag_llm_model_setting(person_id: PersonId, course_id: CourseId):
    """讀取 RAG 出題／批改／弱點分析 LLM 模型（Course_Setting key=llm-model，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_api_key_setting_row(COURSE_SETTING_LLM_MODEL, course_id)
    if not row:
        return RagLlmModelResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return RagLlmModelResponse(
        course_setting_id=row.get("course_setting_id"),
        course_id=course_id,
        llm_model=value or None,
    )


@router.put("/llm-model", response_model=RagLlmModelResponse)
def put_rag_llm_model_setting(
    body: openapi_body(PutRagLlmModelRequest, {"llm_model": "gpt-5.4"}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 RAG 出題／批改／弱點分析 LLM 模型（Course_Setting key=llm-model，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.llm_model or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            COURSE_SETTING_LLM_MODEL,
            value_to_save,
            course_id,
        )
        if not row:
            return RagLlmModelResponse(course_id=course_id, llm_model=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return RagLlmModelResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=course_id,
            llm_model=saved or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
