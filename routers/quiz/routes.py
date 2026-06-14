"""routers.quiz（測驗／Test）routes。

階層 quiz → page（測驗）→ group（自 Bank_Group 快照）→ qa（逐題出題／批改，無追問）。
URL 慣例對齊 bank／exam：建立／列表巢狀於 parent 之下，單一資源以自身主鍵走淺路徑。
出題／批改沿用 bank 的 LLM 管線；金鑰／模型走 quiz- 設定（見 .settings_routes）。
"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path as PathParam
from fastapi.responses import JSONResponse

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.openapi import openapi_body
from utils.serialization import to_json_safe
from utils.taipei_time import now_taipei_iso
from utils.supabase import get_supabase
from utils.bank_storage import generate_page_id

from .schemas import (
    CreateQuizGroupRequest,
    CreateQuizRequest,
    ListQuizAsksResponse,
    ListQuizBankGroupsResponse,
    ListQuizResponse,
    PutQuizGroupAnswerUserPromptTextRequest,
    PutQuizGroupQuestionSystemPromptTextRequest,
    PutQuizGroupQuestionUserPromptTextRequest,
    QuizAskAnswerRateRequest,
    QuizAskRequest,
    QuizGroupAnswerUserPromptTextResponse,
    QuizGroupQuestionSystemPromptTextResponse,
    QuizGroupQuestionUserPromptTextResponse,
    QuizQaAnswerRateRequest,
    QuizQaAnswerRequest,
    QuizQaQuestionRateRequest,
    UpdateQuizGroupRequest,
    UpdateQuizTabNameRequest,
)
from .helpers import (
    _quiz_answer_job_results,
    build_quiz_group_snapshot_row,
    enqueue_quiz_qa_answer_job,
    fetch_bank_group_for_snapshot,
    fetch_bank_unit_for_llm,
    fetch_quiz_ask_row,
    fetch_quiz_group_row,
    fetch_quiz_page_row,
    fetch_quiz_qa_row,
    groups_by_quiz_page_ids,
    list_bank_groups_for_quiz,
    quiz_ask_rows_for_group,
    quiz_llm_ask_impl,
    quiz_llm_generate_qa_impl,
    quiz_llm_regenerate_qa_impl,
    quiz_qa_rows_for_group,
    renumber_quiz_qa_indices,
    require_quiz_group_owner,
)

_logger = logging.getLogger("routers.quiz")

router = APIRouter(prefix="/quiz", tags=["quiz"])


# ---------------------------------------------------------------------------
# 測驗（Quiz）：列表、建立、更名、刪除
# ---------------------------------------------------------------------------


@router.get("/pages", response_model=ListQuizResponse, summary="List Quizzes", operation_id="quiz_list_pages")
def list_quizzes(person_id: PersonId, course_id: CourseId):
    """列出呼叫者的 Quiz（deleted=false、person_id、course_id）。每筆帶 quiz_groups[]（每組帶 qas[]）。"""
    try:
        supabase = get_supabase()
        pid = person_id.strip()
        sel = (
            supabase.table("Quiz")
            .select("*")
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
        quizzes = sel.data or []
        page_ids = list(dict.fromkeys(
            str(q.get("quiz_page_id")) for q in quizzes if q.get("quiz_page_id") is not None
        ))
        groups_by_page = groups_by_quiz_page_ids(page_ids, course_id)
        for q in quizzes:
            q["quiz_groups"] = groups_by_page.get(str(q.get("quiz_page_id") or ""), [])
        quizzes = to_json_safe(quizzes)
        return ListQuizResponse(quizzes=quizzes, count=len(quizzes))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("GET /quiz/pages 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Quiz 失敗: {e!s}")


@router.post("/pages", status_code=201, summary="Create Quiz", operation_id="quiz_create_page")
def create_quiz(
    body: openapi_body(CreateQuizRequest, {"quiz_page_id": "", "person_id": "", "tab_name": "第一回測驗"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """建立一筆 Quiz（測驗）。quiz_page_id 可選（未傳由後端產生）。"""
    fid = (body.quiz_page_id or "").strip()
    body_pid = (body.person_id or "").strip()
    person_id = body_pid if body_pid else caller_person_id
    if body_pid and body_pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與呼叫者（token）不一致")
    if not fid:
        fid = generate_page_id(person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 quiz_page_id")
    tab_name = (body.tab_name or "").strip()

    supabase = get_supabase()
    ts = now_taipei_iso()
    ins = (
        supabase.table("Quiz")
        .insert({
            "quiz_page_id": fid,
            "person_id": person_id,
            "course_id": course_id,
            "tab_name": tab_name,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        })
        .execute()
    )
    if not ins.data:
        raise HTTPException(status_code=500, detail="建立 Quiz 失敗")
    return to_json_safe(ins.data[0])


@router.patch("/pages/{quiz_page_id}", summary="Update Quiz Tab Name", operation_id="quiz_update_page")
def update_quiz_tab_name(
    body: openapi_body(UpdateQuizTabNameRequest, {"tab_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_page_id: str = PathParam(..., description="要更名的 Quiz 的 quiz_page_id"),
):
    """更新既有 Quiz 的 tab_name（以 quiz_page_id 定位；僅 deleted=false）。"""
    fid = (quiz_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 quiz_page_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        row = fetch_quiz_page_row(supabase, fid, course_id, cols="quiz_id, quiz_page_id, person_id")
        if not row:
            raise HTTPException(status_code=404, detail="找不到該 quiz_page_id 的 Quiz 資料，或已刪除")
        if (row.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Quiz")
        ts = now_taipei_iso()
        supabase.table("Quiz").update({"tab_name": tab_name, "updated_at": ts}).eq("quiz_page_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
        return {
            "quiz_id": row.get("quiz_id"),
            "quiz_page_id": fid,
            "tab_name": tab_name,
            "person_id": row.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PATCH /quiz/pages/{quiz_page_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pages/{quiz_page_id}", status_code=200, summary="Delete Quiz", operation_id="quiz_delete_page")
def delete_quiz(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_page_id: str = PathParam(..., description="要刪除的 Quiz 的 quiz_page_id"),
):
    """軟刪除 Quiz（deleted=true；不動其 Quiz_Group／Quiz_QA）。僅 person_id 一致者可刪除。"""
    fid = (quiz_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 quiz_page_id")
    supabase = get_supabase()
    row = fetch_quiz_page_row(supabase, fid, course_id, cols="quiz_id, person_id")
    if not row:
        raise HTTPException(status_code=404, detail="找不到該 quiz_page_id 的 Quiz 資料，或已刪除")
    pid = (row.get("person_id") or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權刪除該 Quiz")
    ts = now_taipei_iso()
    supabase.table("Quiz").update({"deleted": True, "updated_at": ts}).eq("quiz_page_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
    return {"message": "已將 Quiz 標記為刪除", "quiz_page_id": fid, "person_id": pid, "updated_at": ts}


# ---------------------------------------------------------------------------
# 可選用的 Bank 題組（for_exam=true）
# ---------------------------------------------------------------------------


@router.get("/bank-groups", response_model=ListQuizBankGroupsResponse, summary="List Bank Groups for Quiz", operation_id="quiz_list_bank_groups")
def list_quiz_bank_groups(_person_id: PersonId, course_id: CourseId):
    """列出可加入測驗的 Bank_Group（for_exam=true、未刪除），附其題庫 tab_name 與單元 unit_name／unit_type。

    回傳該 course 全部 for_exam 題組，不分建立者。
    """
    groups = to_json_safe(list_bank_groups_for_quiz(course_id))
    return ListQuizBankGroupsResponse(groups=groups, count=len(groups))


# ---------------------------------------------------------------------------
# 題組（Quiz_Group，自 Bank_Group 快照）
# ---------------------------------------------------------------------------


@router.post(
    "/pages/{quiz_page_id}/groups",
    status_code=201,
    summary="Create Quiz Group from Bank Group (snapshot)",
    operation_id="quiz_create_group",
)
def create_quiz_group(
    body: openapi_body(CreateQuizGroupRequest, {"bank_group_id": 1, "group_name": ""}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_page_id: str = PathParam(..., description="目標 Quiz 的 quiz_page_id"),
):
    """挑選一個既有 Bank_Group，把其設定（prompts／qa_count／模型／單元資訊）快照成一筆 Quiz_Group 掛在此測驗下，**不呼叫 LLM**。"""
    try:
        supabase = get_supabase()
        page = fetch_quiz_page_row(supabase, quiz_page_id, course_id, cols="quiz_page_id, person_id")
        if not page:
            raise HTTPException(status_code=404, detail=f"找不到 quiz_page_id={quiz_page_id} 的 Quiz，或已刪除")
        if (page.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權於該 Quiz 新增題組")

        bank_group = fetch_bank_group_for_snapshot(supabase, body.bank_group_id, course_id)
        if not bank_group:
            raise HTTPException(status_code=404, detail=f"找不到 bank_group_id={body.bank_group_id} 的 Bank_Group，或已刪除")

        bank_unit_id = int(bank_group.get("bank_unit_id") or 0)
        bank_unit = fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id) if bank_unit_id > 0 else None

        group_row = build_quiz_group_snapshot_row(
            quiz_page_id=(page.get("quiz_page_id") or quiz_page_id or "").strip(),
            person_id=caller_person_id,
            course_id=course_id,
            bank_group=bank_group,
            bank_unit=bank_unit,
            group_name_override=body.group_name or "",
        )
        ins = supabase.table("Quiz_Group").insert(group_row).execute()
        if not ins.data:
            raise HTTPException(status_code=500, detail="寫入 Quiz_Group 失敗（無回傳資料）")
        return to_json_safe(ins.data[0])
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /quiz/pages/{quiz_page_id}/groups 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/groups/{quiz_group_id}", summary="Get Quiz Group", operation_id="quiz_get_group")
def get_quiz_group(
    _caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """讀取單一題組（含其 Quiz_QA，依 question_series_index 升序）。"""
    supabase = get_supabase()
    group = fetch_quiz_group_row(supabase, quiz_group_id, course_id, cols="*")
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 quiz_group_id={quiz_group_id} 的 Quiz_Group，或已刪除")
    group = to_json_safe(group)
    group["qas"] = to_json_safe(quiz_qa_rows_for_group(supabase, quiz_group_id, course_id))
    return group


def _get_quiz_group_prompt_field(quiz_group_id: int, course_id: int, field: str) -> dict[str, Any]:
    supabase = get_supabase()
    cols = f"quiz_group_id, {field}"
    group = fetch_quiz_group_row(supabase, quiz_group_id, course_id, cols=cols)
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 quiz_group_id={quiz_group_id} 的 Quiz_Group，或已刪除")
    return {"quiz_group_id": quiz_group_id, field: group.get(field) or ""}


def _put_quiz_group_prompt_field(
    *,
    quiz_group_id: int,
    course_id: int,
    caller_person_id: str,
    field: str,
    value: str,
) -> dict[str, Any]:
    supabase = get_supabase()
    require_quiz_group_owner(
        supabase, quiz_group_id, course_id, caller_person_id, cols="quiz_group_id, person_id, course_id"
    )
    supabase.table("Quiz_Group").update({
        field: value,
        "updated_at": now_taipei_iso(),
    }).eq("quiz_group_id", quiz_group_id).eq("deleted", False).execute()
    return _get_quiz_group_prompt_field(quiz_group_id, course_id, field)


@router.get(
    "/groups/{quiz_group_id}/question-system-prompt-text",
    summary="Get Quiz Group question_system_prompt_text",
    operation_id="quiz_get_group_question_system_prompt_text",
)
def get_quiz_group_question_system_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """讀取 Quiz_Group.question_system_prompt_text。"""
    return QuizGroupQuestionSystemPromptTextResponse(**_get_quiz_group_prompt_field(
        quiz_group_id, course_id, "question_system_prompt_text"
    ))


@router.put(
    "/groups/{quiz_group_id}/question-system-prompt-text",
    summary="Update Quiz Group question_system_prompt_text",
    operation_id="quiz_put_group_question_system_prompt_text",
)
def put_quiz_group_question_system_prompt_text(
    body: openapi_body(
        PutQuizGroupQuestionSystemPromptTextRequest,
        {"question_system_prompt_text": ""},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """寫入 Quiz_Group.question_system_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return QuizGroupQuestionSystemPromptTextResponse(**_put_quiz_group_prompt_field(
            quiz_group_id=quiz_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="question_system_prompt_text",
            value=body.question_system_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /quiz/groups/{quiz_group_id}/question-system-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/groups/{quiz_group_id}/question-user-prompt-text",
    summary="Get Quiz Group question_user_prompt_text",
    operation_id="quiz_get_group_question_user_prompt_text",
)
def get_quiz_group_question_user_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """讀取 Quiz_Group.question_user_prompt_text。"""
    return QuizGroupQuestionUserPromptTextResponse(**_get_quiz_group_prompt_field(
        quiz_group_id, course_id, "question_user_prompt_text"
    ))


@router.put(
    "/groups/{quiz_group_id}/question-user-prompt-text",
    summary="Update Quiz Group question_user_prompt_text",
    operation_id="quiz_put_group_question_user_prompt_text",
)
def put_quiz_group_question_user_prompt_text(
    body: openapi_body(
        PutQuizGroupQuestionUserPromptTextRequest,
        {"question_user_prompt_text": ""},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """寫入 Quiz_Group.question_user_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return QuizGroupQuestionUserPromptTextResponse(**_put_quiz_group_prompt_field(
            quiz_group_id=quiz_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="question_user_prompt_text",
            value=body.question_user_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /quiz/groups/{quiz_group_id}/question-user-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/groups/{quiz_group_id}/answer-user-prompt-text",
    summary="Get Quiz Group answer_user_prompt_text",
    operation_id="quiz_get_group_answer_user_prompt_text",
)
def get_quiz_group_answer_user_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """讀取 Quiz_Group.answer_user_prompt_text。"""
    return QuizGroupAnswerUserPromptTextResponse(**_get_quiz_group_prompt_field(
        quiz_group_id, course_id, "answer_user_prompt_text"
    ))


@router.put(
    "/groups/{quiz_group_id}/answer-user-prompt-text",
    summary="Update Quiz Group answer_user_prompt_text",
    operation_id="quiz_put_group_answer_user_prompt_text",
)
def put_quiz_group_answer_user_prompt_text(
    body: openapi_body(
        PutQuizGroupAnswerUserPromptTextRequest,
        {"answer_user_prompt_text": ""},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """寫入 Quiz_Group.answer_user_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return QuizGroupAnswerUserPromptTextResponse(**_put_quiz_group_prompt_field(
            quiz_group_id=quiz_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="answer_user_prompt_text",
            value=body.answer_user_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /quiz/groups/{quiz_group_id}/answer-user-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/groups/{quiz_group_id}", summary="Update Quiz Group", operation_id="quiz_update_group")
def update_quiz_group(
    body: openapi_body(
        UpdateQuizGroupRequest,
        {
            "group_name": "新名稱",
            "qa_count": 8,
            "question_system_prompt_text": "",
            "question_user_prompt_text": "",
            "question_llm_model": "",
            "answer_user_prompt_text": "",
            "answer_llm_model": "",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """更新題組快照（僅更新有傳入的欄位）。僅 person_id 一致者可更新。"""
    try:
        supabase = get_supabase()
        require_quiz_group_owner(
            supabase, quiz_group_id, course_id, caller_person_id, cols="quiz_group_id, person_id, course_id"
        )

        update_payload: dict[str, Any] = {}
        if body.group_name is not None:
            update_payload["group_name"] = body.group_name.strip()
        if body.qa_count is not None:
            update_payload["qa_count"] = int(body.qa_count)
        if "question_system_prompt_text" in body.model_fields_set:
            update_payload["question_system_prompt_text"] = body.question_system_prompt_text
        if "question_user_prompt_text" in body.model_fields_set:
            update_payload["question_user_prompt_text"] = body.question_user_prompt_text
        if body.question_llm_model is not None:
            update_payload["question_llm_model"] = body.question_llm_model.strip()
        if "answer_user_prompt_text" in body.model_fields_set:
            update_payload["answer_user_prompt_text"] = body.answer_user_prompt_text
        if body.answer_llm_model is not None:
            update_payload["answer_llm_model"] = body.answer_llm_model.strip()
        if not update_payload:
            raise HTTPException(status_code=400, detail="未提供任何要更新的欄位")
        update_payload["updated_at"] = now_taipei_iso()

        supabase.table("Quiz_Group").update(update_payload).eq("quiz_group_id", quiz_group_id).eq("deleted", False).execute()
        read = fetch_quiz_group_row(supabase, quiz_group_id, course_id, cols="*")
        return to_json_safe(read or {})
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PATCH /quiz/groups/{quiz_group_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/groups/{quiz_group_id}", status_code=200, summary="Delete Quiz Group", operation_id="quiz_delete_group")
def delete_quiz_group(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """軟刪除題組（僅將此題組 deleted=true，不動其 Quiz_QA）。僅 person_id 一致者可刪除。"""
    try:
        supabase = get_supabase()
        group = require_quiz_group_owner(
            supabase, quiz_group_id, course_id, caller_person_id, cols="quiz_group_id, person_id, course_id"
        )
        ts = now_taipei_iso()
        supabase.table("Quiz_Group").update({"deleted": True, "updated_at": ts}).eq("quiz_group_id", quiz_group_id).eq("deleted", False).execute()
        return {
            "message": "已將 Quiz_Group 標記為刪除",
            "quiz_group_id": quiz_group_id,
            "person_id": group.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /quiz/groups/{quiz_group_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 題組內逐題出題（qa 為 group 的子集合；無追問）
# ---------------------------------------------------------------------------


@router.post("/groups/{quiz_group_id}/qa/llm-generate", summary="Quiz LLM Generate Next QA", operation_id="quiz_llm_generate_qa")
def quiz_llm_generate_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """
    在題組內產生**下一題**（LLM，同步）。一律自 Quiz_Group 讀取 question_system_prompt_text／question_user_prompt_text；
    同題組既有題目題幹作為「已出過題目（勿重複）」送入；並併入本題組 Quiz_Ask 追問紀錄作為出題依據。
    已達 qa_count 上限時回 409。出題成功後新增一筆 Quiz_QA。無 request body。
    """
    return quiz_llm_generate_qa_impl(
        quiz_group_id=quiz_group_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )


@router.post("/qa/{quiz_qa_id}/llm-regenerate", summary="Quiz LLM Regenerate QA (in place)", operation_id="quiz_llm_regenerate_qa")
def quiz_llm_regenerate_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_qa_id: int = PathParam(..., gt=0, description="Quiz_QA 主鍵"),
):
    """
    **原地重出同一題**（LLM，同步）：只重新產生這一題的 question_* 內容並覆寫回同一 quiz_qa_id，
    prompt 一律自所屬 Quiz_Group 讀取；並併入本題組 Quiz_Ask 追問紀錄作為出題依據。
    不刪除、不新增、不改 question_series_index。同題組此題之前的題作為「勿重複」送入。
    重出後本題舊作答／批改／評分清空。不檢查 qa_count 上限。無 request body。
    """
    return quiz_llm_regenerate_qa_impl(
        quiz_qa_id=quiz_qa_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )


# ---------------------------------------------------------------------------
# 追問（對題組對應之 Bank 課程內容發問）
# ---------------------------------------------------------------------------


@router.post("/groups/{quiz_group_id}/llm-ask", summary="Quiz LLM Ask (course content)", operation_id="quiz_llm_ask")
def quiz_llm_ask(
    body: openapi_body(QuizAskRequest, {"ask_user_prompt_text": ""}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """
    出題後，對該題組對應 **Bank 單元的課程內容**發問（LLM，同步）。
    prompt 含題組出題規定、本題組全部測驗題紀錄（題目／提示／參考答案／作答／評閱）、
    先前追問紀錄與本次提問；unit_type 2／3／4 以 transcript 純 LLM 回答，其餘載入 RAG ZIP 檢索。
    每次呼叫於 public.Quiz_Ask 新增一列並回傳（含 answer_content、ask_llm_model）。LLM 失敗回 200 + llm_error。
    """
    return quiz_llm_ask_impl(
        quiz_group_id=quiz_group_id,
        ask_user_prompt_text=body.ask_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )


@router.get(
    "/groups/{quiz_group_id}/asks",
    response_model=ListQuizAsksResponse,
    summary="List Quiz Group Asks",
    operation_id="quiz_list_asks",
)
def list_quiz_group_asks(
    _caller_person_id: PersonId,
    course_id: CourseId,
    quiz_group_id: int = PathParam(..., gt=0, description="Quiz_Group 主鍵"),
):
    """列出該題組歷次提問（Quiz_Ask，依 created_at 由舊到新）。"""
    supabase = get_supabase()
    group = fetch_quiz_group_row(supabase, quiz_group_id, course_id, cols="quiz_group_id")
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 quiz_group_id={quiz_group_id} 的 Quiz_Group，或已刪除")
    asks = to_json_safe(quiz_ask_rows_for_group(supabase, quiz_group_id, course_id))
    return ListQuizAsksResponse(asks=asks, count=len(asks))


@router.put("/asks/{quiz_ask_id}/answer-rate", status_code=200, summary="Quiz Rate Ask Answer", operation_id="quiz_ask_answer_rate")
def update_quiz_ask_answer_rate(
    body: openapi_body(QuizAskAnswerRateRequest, {"answer_rate": 1}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_ask_id: int = PathParam(..., gt=0, description="Quiz_Ask 主鍵"),
):
    """更新 Quiz_Ask.answer_rate（-1／0／1）。僅 person_id 一致者可更新。"""
    try:
        supabase = get_supabase()
        ask = fetch_quiz_ask_row(supabase, quiz_ask_id, course_id, cols="quiz_ask_id, person_id")
        if not ask:
            raise HTTPException(status_code=404, detail=f"找不到 quiz_ask_id={quiz_ask_id} 的 Quiz_Ask，或已刪除")
        if (ask.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權評分該 Quiz_Ask")
        ts = now_taipei_iso()
        supabase.table("Quiz_Ask").update({"answer_rate": int(body.answer_rate), "updated_at": ts}).eq("quiz_ask_id", quiz_ask_id).eq("deleted", False).execute()
        return {"quiz_ask_id": quiz_ask_id, "answer_rate": int(body.answer_rate), "updated_at": ts}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /quiz/asks/{quiz_ask_id}/answer-rate 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/asks/{quiz_ask_id}", status_code=200, summary="Delete Quiz Ask", operation_id="quiz_ask_delete")
def delete_quiz_ask(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_ask_id: int = PathParam(..., gt=0, description="Quiz_Ask 主鍵"),
):
    """軟刪除單筆提問（Quiz_Ask.deleted=true）。僅 person_id 一致者可刪除。"""
    try:
        supabase = get_supabase()
        ask = fetch_quiz_ask_row(supabase, quiz_ask_id, course_id, cols="quiz_ask_id, quiz_group_id, person_id")
        if not ask:
            raise HTTPException(status_code=404, detail=f"找不到 quiz_ask_id={quiz_ask_id} 的 Quiz_Ask，或已刪除")
        if (ask.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Quiz_Ask")
        ts = now_taipei_iso()
        supabase.table("Quiz_Ask").update({"deleted": True, "updated_at": ts}).eq("quiz_ask_id", quiz_ask_id).eq("deleted", False).execute()
        return {
            "message": "已將 Quiz_Ask 標記為刪除",
            "quiz_ask_id": quiz_ask_id,
            "quiz_group_id": ask.get("quiz_group_id"),
            "person_id": ask.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /quiz/asks/{quiz_ask_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 批改（非同步）與評分、刪除（以主鍵淺路徑引用）
# ---------------------------------------------------------------------------


@router.post("/qa/{quiz_qa_id}/llm-answer", summary="Quiz Answer QA", operation_id="quiz_llm_answer_qa")
async def quiz_qa_answer(
    background_tasks: BackgroundTasks,
    body: openapi_body(QuizQaAnswerRequest, {"answer_content": "學生作答文字"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_qa_id: int = PathParam(..., gt=0, description="Quiz_QA 主鍵"),
):
    """
    非同步批改：以路徑 quiz_qa_id 指定要批改的題目，使用所屬題組之 answer_user_prompt_text 批改學生作答。
    unit_type 2／3／4 以 transcript 純 LLM 批改；其餘載入該單元 RAG ZIP。
    回傳 202 + job_id；輪詢 GET /quiz/qa/answer-result/{job_id}。
    """
    return await enqueue_quiz_qa_answer_job(
        background_tasks,
        caller_person_id,
        course_id,
        quiz_qa_id=quiz_qa_id,
        answer_content=body.answer_content,
    )


@router.get("/qa/answer-result/{job_id}", summary="Get Quiz QA Answer Result", operation_id="quiz_qa_answer_result")
async def get_quiz_qa_answer_result(_person_id: PersonId, course_id: CourseId, job_id: str):
    """輪詢批改結果。status: pending | ready | error；ready 時回讀整列 Quiz_QA。"""
    if job_id not in _quiz_answer_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出批改）",
            },
        )
    data = _quiz_answer_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "llm_error": data.get("llm_error"),
    }
    quiz_qa_row: dict[str, Any] | None = None
    if data["status"] == "ready":
        rid = data.get("quiz_qa_id")
        try:
            rid_int = int(rid) if rid is not None else 0
        except (TypeError, ValueError):
            rid_int = 0
        if rid_int > 0:
            try:
                supabase = get_supabase()
                quiz_qa_row = to_json_safe(fetch_quiz_qa_row(supabase, rid_int, course_id, cols="*") or {}) or None
            except Exception as e:
                _logger.warning("answer-result 讀取 Quiz_QA 失敗 job_id=%s: %s", job_id, e)
        out["quiz_qa"] = quiz_qa_row
    return out


@router.put("/qa/{quiz_qa_id}/question-rate", status_code=200, summary="Quiz Rate Question", operation_id="quiz_qa_question_rate")
def update_quiz_qa_question_rate(
    body: openapi_body(QuizQaQuestionRateRequest, {"question_rate": 1}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_qa_id: int = PathParam(..., gt=0, description="Quiz_QA 主鍵"),
):
    """更新 Quiz_QA.question_rate（-1／0／1）。僅 person_id 一致者可更新。"""
    return _update_quiz_qa_rate(
        caller_person_id, course_id, quiz_qa_id, field="question_rate", value=body.question_rate
    )


@router.put("/qa/{quiz_qa_id}/answer-rate", status_code=200, summary="Quiz Rate Answer", operation_id="quiz_qa_answer_rate")
def update_quiz_qa_answer_rate(
    body: openapi_body(QuizQaAnswerRateRequest, {"answer_rate": 1}),
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_qa_id: int = PathParam(..., gt=0, description="Quiz_QA 主鍵"),
):
    """更新 Quiz_QA.answer_rate（-1／0／1）。僅 person_id 一致者可更新。"""
    return _update_quiz_qa_rate(
        caller_person_id, course_id, quiz_qa_id, field="answer_rate", value=body.answer_rate
    )


def _update_quiz_qa_rate(caller_person_id: str, course_id: int, quiz_qa_id: int, *, field: str, value: int) -> dict:
    """共用：更新 Quiz_QA 的 question_rate／answer_rate。"""
    try:
        supabase = get_supabase()
        qa = fetch_quiz_qa_row(supabase, quiz_qa_id, course_id, cols="quiz_qa_id, person_id")
        if not qa:
            raise HTTPException(status_code=404, detail=f"找不到 quiz_qa_id={quiz_qa_id} 的 Quiz_QA，或已刪除")
        if (qa.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權評分該 Quiz_QA")
        ts = now_taipei_iso()
        supabase.table("Quiz_QA").update({field: int(value), "updated_at": ts}).eq("quiz_qa_id", quiz_qa_id).eq("deleted", False).execute()
        return {"quiz_qa_id": quiz_qa_id, field: int(value), "updated_at": ts}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /quiz/qa/{quiz_qa_id}/%s 錯誤", field)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/qa/{quiz_qa_id}", status_code=200, summary="Delete Quiz QA", operation_id="quiz_qa_delete")
def delete_quiz_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    quiz_qa_id: int = PathParam(..., gt=0, description="Quiz_QA 主鍵"),
):
    """軟刪除單一題目（Quiz_QA.deleted=true）。僅 person_id 一致者可刪除。"""
    try:
        supabase = get_supabase()
        qa = fetch_quiz_qa_row(supabase, quiz_qa_id, course_id, cols="quiz_qa_id, quiz_group_id, person_id")
        if not qa:
            raise HTTPException(status_code=404, detail=f"找不到 quiz_qa_id={quiz_qa_id} 的 Quiz_QA，或已刪除")
        if (qa.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Quiz_QA")
        ts = now_taipei_iso()
        supabase.table("Quiz_QA").update({"deleted": True, "updated_at": ts}).eq("quiz_qa_id", quiz_qa_id).eq("deleted", False).execute()
        quiz_group_id_val = qa.get("quiz_group_id")
        if quiz_group_id_val:
            renumber_quiz_qa_indices(supabase, int(quiz_group_id_val), course_id)
        return {
            "message": "已將 Quiz_QA 標記為刪除",
            "quiz_qa_id": quiz_qa_id,
            "quiz_group_id": quiz_group_id_val,
            "person_id": qa.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /quiz/qa/{quiz_qa_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))
