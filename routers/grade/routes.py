"""routers.grade routes（自 grade.py 拆分）。"""

import base64
import logging
from pathlib import Path
from typing import Annotated, Any

from utils.openapi import openapi_body

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from postgrest.exceptions import APIError
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
from utils.media import (
    audio_media_type_for_suffix,
)
from utils.rag_stem import transcript_from_row
from utils.rag_transcript import (
    pick_audio_from_upload_zip,
    read_mp3_unit_transcript_from_upload_zip,
    read_supplementary_text_from_youtube_unit,
    read_youtube_video_id_from_upload_zip,
)
from utils.rag_course import (
    execute_with_course_id_fallback,
    require_rag_tab_owner,
    resolve_rag_tab_owner_person_id,
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
    QuizGradeDbOnlyRequest,
    QuizGradeRequest,
    RagApiKeyExistsResponse,
    RagApiKeyResponse,
    RagLlmModelResponse,
    RagQuizFollowupRequest,
    RagQuizForExamRequest,
    RagUnitMp3FileFromZipResponse,
    RagUnitTextResponse,
    RagUnitYoutubeUrlFromZipResponse,
)
from .helpers import (
    _enqueue_rag_llm_grade_job,
    _grade_job_results,
    _quiz_history_prompt_dicts,
    _rag_llm_generate_quiz_impl,
    _read_upload_zip_bytes_or_http_error,
    _transcript_from_upload_zip_for_folder,
)

_logger = logging.getLogger("routers.grade")


router = APIRouter(prefix="/rag", tags=["rag"])

RAG_UNIT_TYPE_TEXT = 2


# ---------------------------------------------------------------------------
# POST /rag/page/unit/quiz/followup
# ---------------------------------------------------------------------------


@router.post("/page/unit/quiz/followup", summary="Set Rag Quiz follow_up flag", operation_id="rag_quiz_followup")
def mark_rag_quiz_followup(
    body: openapi_body(
        RagQuizFollowupRequest,
        {"rag_quiz_id": 1, "rag_page_id": "", "rag_unit_id": 0, "followup": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.follow_up（followup=true 標記追問、false 取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
    req_tab = (body.rag_page_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
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
                .eq("rag_quiz_id", body.rag_quiz_id)
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
        if req_tab and (row0.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_quiz_id 對應資料不一致")
        if req_unit > 0 and int(row0.get("rag_unit_id") or 0) != req_unit:
            raise HTTPException(status_code=400, detail="rag_unit_id 與 rag_quiz_id 對應資料不一致")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update(
            {"follow_up": body.followup, "updated_at": ts}
        ).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", body.rag_quiz_id)
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
        _logger.exception("POST /rag/page/unit/quiz/followup 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /rag/page/unit/quiz/llm-generate
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


@router.post("/page/unit/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
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
# POST /rag/page/unit/quiz/llm-generate-followup
# ---------------------------------------------------------------------------


@router.post(
    "/page/unit/quiz/llm-generate-followup",
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
# POST /rag/page/unit/quiz/llm-generate-db
# ---------------------------------------------------------------------------


@router.post(
    "/page/unit/quiz/llm-generate-db",
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
# POST /rag/page/unit/quiz/llm-generate-followup-db
# ---------------------------------------------------------------------------


@router.post(
    "/page/unit/quiz/llm-generate-followup-db",
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


@router.post("/page/unit/quiz/llm-grade", summary="Rag Grade Quiz")
async def grade_submission(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizGradeRequest,
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
    回傳 202 + job_id；輪詢 GET /rag/page/unit/quiz/grade-result/{job_id}。
    """
    return await _enqueue_rag_llm_grade_job(
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
    "/page/unit/quiz/llm-grade-db",
    summary="Rag Grade Quiz (stored answer_user_prompt_text)",
    operation_id="rag_llm_grade_quiz_db_prompt",
)
async def grade_submission_stored_answer_prompt(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizGradeDbOnlyRequest,
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
    與 `llm-grade` 相同，但請求不含 `answer_user_prompt_text`；
    評分時與寫回皆以 Rag_Quiz 該列既有之 `answer_user_prompt_text` 為準。
    """
    return await _enqueue_rag_llm_grade_job(
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
# POST /rag/page/unit/quiz/for-exam
# ---------------------------------------------------------------------------

@router.post("/page/unit/quiz/for-exam", summary="Set Rag Quiz for_exam flag")
def mark_rag_quiz_for_exam(
    body: openapi_body(
        RagQuizForExamRequest,
        {"rag_quiz_id": 1, "rag_page_id": "", "rag_unit_id": 0, "for_exam": True},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.for_exam（true＝測驗用、false＝取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
    req_tab = (body.rag_page_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
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
                .eq("rag_quiz_id", body.rag_quiz_id)
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
        if req_tab and (row0.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_quiz_id 對應資料不一致")
        if req_unit > 0 and int(row0.get("rag_unit_id") or 0) != req_unit:
            raise HTTPException(status_code=400, detail="rag_unit_id 與 rag_quiz_id 對應資料不一致")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"for_exam": body.for_exam, "updated_at": ts}).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", body.rag_quiz_id)
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
        _logger.exception("POST /rag/page/unit/quiz/for-exam 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /rag/page/unit/quiz/grade-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/page/unit/quiz/grade-result/{job_id}", summary="Get Grade Result", tags=["rag"])
async def get_grade_result(job_id: str, _person_id: PersonId, course_id: CourseId):
    """
    輪詢評分結果。status: pending | ready | error；
    ready 時 result 為 quiz_comments、rag_quiz_id（另含 rag_answer_id），並自資料庫讀取 rag_quiz 整列。
    """
    if job_id not in _grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _grade_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
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

                        def build_grade_result_sel(with_course_filter: bool):
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
                            "Rag_Quiz", build_grade_result_sel, course_id
                        )
                        if q.data:
                            rag_quiz_row = to_json_safe(q.data[0])
                            if isinstance(rag_quiz_row, dict):
                                rag_quiz_row["quiz_history_list"] = parse_rag_quiz_history_list(
                                    rag_quiz_row.get("quiz_history_list")
                                )
                except (TypeError, ValueError) as e:
                    _logger.debug("grade-result rag_quiz_id 無效 job_id=%s: %s", job_id, e)
                except Exception as e:
                    _logger.warning("grade-result 讀取 Rag_Quiz 失敗 job_id=%s: %s", job_id, e)
        out["rag_quiz"] = rag_quiz_row
    return out


@router.get(
    "/unit/text",
    summary="Rag Unit Text",
    operation_id="rag_unit_text",
    response_model=RagUnitTextResponse,
)
def rag_unit_text(
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab）"),
    folder_name: str = Query(
        "",
        description="與 upload ZIP 內單元資料夾名相同；與 rag_unit_id 二擇一（有 folder_name 時須傳 person_id）",
    ),
    rag_unit_id: int = Query(
        0,
        ge=0,
        description="Rag_Unit 主鍵；與 folder_name 二擇一",
    ),
    person_id: Annotated[
        str | None,
        Query(
            alias="person_id",
            description="使用 folder_name 時必填；僅 rag_unit_id 時可不傳",
        ),
    ] = None,
):
    """
    回傳 **unit_type=2（文字單元）** 之 `text_file_name` 與 `transcript`（全文，含 Markdown）。

    - **folder_name**：自 upload ZIP 讀取（與 build-rag-zip unit_type=2 一致）；須傳 `person_id`。
    - **rag_unit_id**：自 `Rag_Unit` 讀取，**不需** `person_id`；若 DB 無逐字稿則改讀 upload ZIP（以 `folder_combination` 或 `unit_name` 為資料夾名）。
    """
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    unit_id = int(rag_unit_id or 0)

    if folder and unit_id > 0:
        raise HTTPException(status_code=400, detail="folder_name 與 rag_unit_id 請二擇一")
    if not folder and unit_id <= 0:
        raise HTTPException(status_code=400, detail="請傳入 folder_name 或 rag_unit_id（二擇一）")

    if folder:
        pid = (person_id or "").strip()
        if not pid:
            raise HTTPException(status_code=400, detail="使用 folder_name 時須傳入 person_id")
        require_rag_tab_owner(pid, rag_page_id, course_id)
        try:
            transcript, text_file_name = _transcript_from_upload_zip_for_folder(pid, tab, folder)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _logger.exception("GET /rag/unit/text 讀取 upload ZIP 失敗")
            raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e
        return RagUnitTextResponse(
            rag_page_id=tab,
            folder_name=folder,
            rag_unit_id=0,
            text_file_name=text_file_name,
            transcript=transcript,
        )

    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    supabase = get_supabase()

    def build_text_sel(with_course_filter: bool, *, include_folder: bool):
        cols = (
            "rag_unit_id, rag_page_id, unit_type, unit_name, folder_combination, "
            "text_file_name, transcript, deleted, course_id"
            if include_folder
            else "rag_unit_id, rag_page_id, unit_type, unit_name, text_file_name, transcript, deleted, course_id"
        )
        cols = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
        q = (
            supabase.table("Rag_Unit")
            .select(cols)
            .eq("rag_unit_id", unit_id)
            .eq("person_id", owner_pid)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    try:
        sel = execute_with_course_id_fallback(
            "Rag_Unit",
            lambda wc: build_text_sel(wc, include_folder=True),
            course_id,
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            try:
                sel = execute_with_course_id_fallback(
                    "Rag_Unit",
                    lambda wc: build_text_sel(wc, include_folder=False),
                    course_id,
                )
            except Exception as e2:
                _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
                raise HTTPException(status_code=500, detail=f"查詢失敗: {e2!s}") from e2
        else:
            _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
            raise HTTPException(status_code=500, detail=f"查詢失敗: {e!s}") from e
    except Exception as e:
        _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
        raise HTTPException(status_code=500, detail=f"查詢失敗: {e!s}") from e

    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_unit_id，或與此 rag_page_id／擁有者不一致",
        )
    row = sel.data[0]
    if row.get("deleted"):
        raise HTTPException(status_code=404, detail="該單元已刪除")
    if (row.get("rag_page_id") or "").strip() != tab:
        raise HTTPException(
            status_code=400,
            detail="rag_page_id 與該 rag_unit_id 所屬之 Rag_Unit.rag_page_id 不一致",
        )
    try:
        ut = int(row.get("unit_type") or 0)
    except (TypeError, ValueError):
        ut = 0
    if ut != RAG_UNIT_TYPE_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"僅 unit_type=2（文字單元）可使用此端點，目前 unit_type={ut}",
        )

    text_file_name = (row.get("text_file_name") or "").strip()
    transcript = transcript_from_row(row)

    zip_folder = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if not transcript and zip_folder:
        try:
            transcript, zip_text_name = _transcript_from_upload_zip_for_folder(
                owner_pid, tab, zip_folder
            )
            if not text_file_name:
                text_file_name = zip_text_name
        except (FileNotFoundError, ValueError) as e:
            _logger.debug("GET /rag/unit/text ZIP 備援略過: %s", e)
        except Exception:
            _logger.exception("GET /rag/unit/text ZIP 備援失敗")

    return RagUnitTextResponse(
        rag_page_id=tab,
        folder_name=zip_folder,
        rag_unit_id=unit_id,
        text_file_name=text_file_name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/unit/mp3-file
# ---------------------------------------------------------------------------


@router.get(
    "/unit/mp3-file",
    summary="Rag Unit Audio File",
    operation_id="rag_unit_mp3_file",
    response_model=RagUnitMp3FileFromZipResponse,
)
def rag_unit_audio_file(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同",
    ),
):
    """
    自 upload ZIP 內指定資料夾擷取音訊（base64）與**恰好一個**文字檔全文作為 `transcript`（與 build-rag-zip unit_type=3 一致；須音訊＋逐字稿）。
    query 須含 `person_id`，且須與該 `rag_page_id` 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        transcript, text_file_name = read_mp3_unit_transcript_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    media = audio_media_type_for_suffix(suffix)
    disp_name = Path(inner_path).name
    return RagUnitMp3FileFromZipResponse(
        rag_page_id=tab,
        folder_name=folder,
        audio_base64=base64.b64encode(contents).decode(),
        media_type=media,
        filename=disp_name,
        text_file_name=text_file_name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/unit/youtube-url
# ---------------------------------------------------------------------------


@router.get(
    "/unit/youtube-url",
    summary="Rag Unit Youtube Url",
    operation_id="rag_unit_youtube_url",
    response_model=RagUnitYoutubeUrlFromZipResponse,
)
def rag_unit_youtube_url(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同",
    ),
):
    """
    自 upload ZIP 內指定資料夾讀取**恰好一個**文字檔：第一行為 YouTube URL，第二行起為 `transcript`（與 build-rag-zip unit_type=4 一致）。
    query 須含 `person_id`，且須與該 rag_page_id 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        vid, inner_path = read_youtube_video_id_from_upload_zip(zip_bytes, folder)
        transcript, _ = read_supplementary_text_from_youtube_unit(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RagUnitYoutubeUrlFromZipResponse(
        rag_page_id=tab,
        folder_name=folder,
        youtube_url=f"https://www.youtube.com/watch?v={vid}",
        text_file_name=Path(inner_path).name,
        transcript=transcript,
    )


@router.get("/llm_api_key/exists", response_model=RagApiKeyExistsResponse)
def get_rag_api_key_exists(person_id: PersonId, course_id: CourseId):
    """查詢 RAG LLM API Key 是否已設定（Course_Setting key=rag-api-key，依 course_id）；不回傳 key 內容。"""
    _require_active_person(person_id)
    return RagApiKeyExistsResponse(
        course_id=course_id,
        exists=course_api_key_exists(COURSE_SETTING_RAG_API_KEY, course_id),
    )


@router.get("/llm_api_key", response_model=RagApiKeyResponse)
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


@router.put("/llm_api_key", response_model=RagApiKeyResponse)
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


@router.get("/llm_model", response_model=RagLlmModelResponse)
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


@router.put("/llm_model", response_model=RagLlmModelResponse)
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
