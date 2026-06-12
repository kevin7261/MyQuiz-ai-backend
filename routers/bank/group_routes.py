"""routers.bank 題組（Bank_Group）／問答（Bank_QA）routes。

階層關係 bank → page → unit → group → qa（與 rag 的資料模型一致）。
URL 採業界常見慣例（與 rag 一致）：**建立／列表**巢狀於 parent 之下以圈定集合，
**單一資源**則以自身唯一主鍵走淺路徑引用，不強迫帶完整祖先。

- 建題組／列題組：POST／GET /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups
- 單一題組：GET／PATCH／DELETE /bank/groups/{bank_group_id}、PUT /bank/groups/{bank_group_id}/for-exam
- 題組內出題：POST /bank/groups/{bank_group_id}/qa/llm-generate（上限 qa_count）
- 單題批改：POST /bank/qa/{bank_qa_id}/llm-answer → GET /bank/qa/answer-result/{job_id}
- 刪單題：DELETE /bank/qa/{bank_qa_id}

對應 rag 的 Rag_Quiz 出題／批改，但以「題組」為單位、無追問。
LLM API Key／模型用 bank 專屬課程設定（/v1/bank/llm-api-key、/v1/bank/llm-model；Course_Setting key=bank-api-key／bank-llm-model），與 rag、exam 完全分開；題組亦可指定 question_llm_model／answer_llm_model。
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
from utils.bank_course import execute_with_course_id_fallback, insert_bank_child_row

from utils.course_setting import (
    COURSE_SETTING_BANK_ANSWER_USER_PROMPT_TEXT,
    COURSE_SETTING_BANK_QUESTION_SYSTEM_PROMPT_TEXT,
    COURSE_SETTING_BANK_QUESTION_USER_PROMPT_TEXT,
    resolve_group_prompt_texts,
)

from .group_schemas import (
    BankGroupAnswerUserPromptTextResponse,
    BankGroupForExamRequest,
    BankGroupQuestionSystemPromptTextResponse,
    BankGroupQuestionUserPromptTextResponse,
    BankQaAnswerRequest,
    CreateBankGroupRequest,
    PutBankGroupAnswerUserPromptTextRequest,
    PutBankGroupQuestionSystemPromptTextRequest,
    PutBankGroupQuestionUserPromptTextRequest,
    UpdateBankGroupRequest,
)
from .group_helpers import (
    _bank_answer_job_results,
    _bank_qa_rows_for_group,
    _fetch_bank_group_row,
    _fetch_bank_qa_row,
    bank_llm_generate_qa_impl,
    bank_llm_regenerate_qa_impl,
    enqueue_bank_qa_answer_job,
    fetch_bank_unit_in_page,
    groups_by_bank_unit_ids,
    require_bank_group_owner,
)

_logger = logging.getLogger("routers.bank")

router = APIRouter(prefix="/bank", tags=["bank"])


# ---------------------------------------------------------------------------
# 題組集合（巢狀於 page/unit）：建立、列表
# ---------------------------------------------------------------------------


@router.post(
    "/pages/{bank_page_id}/units/{bank_unit_id}/groups",
    status_code=201,
    summary="Bank Create Group (no LLM)",
    operation_id="bank_create_group",
)
def create_bank_group(
    body: openapi_body(
        CreateBankGroupRequest,
        {
            "group_name": "第一回測驗",
            "qa_count": 5,
            "question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。",
            "question_user_prompt_text": "請就課程內容出一道問答題。",
            "question_llm_model": "",
            "answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。",
            "answer_llm_model": "",
            "for_exam": False,
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_page_id: str = PathParam(..., description="Bank.bank_page_id"),
    bank_unit_id: int = PathParam(..., gt=0, description="Bank_Unit 主鍵"),
):
    """
    在指定單元下新增一個測試題組（Bank_Group），**不呼叫 LLM**。
    設定 `qa_count`（題數上限）與出題／批改 prompt 後，再以 POST /bank/groups/{id}/qa/llm-generate 逐題產生。
    """
    try:
        supabase = get_supabase()
        unit = fetch_bank_unit_in_page(
            supabase, bank_page_id=bank_page_id, bank_unit_id=bank_unit_id, course_id=course_id
        )
        pid = (unit.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權於該 Bank_Unit 新增題組")
        page_id = (unit.get("bank_page_id") or bank_page_id or "").strip()

        ts = now_taipei_iso()
        prompt_texts = resolve_group_prompt_texts(
            body_system=body.question_system_prompt_text,
            body_user=body.question_user_prompt_text,
            body_answer=body.answer_user_prompt_text,
            course_id=course_id,
            system_key=COURSE_SETTING_BANK_QUESTION_SYSTEM_PROMPT_TEXT,
            user_key=COURSE_SETTING_BANK_QUESTION_USER_PROMPT_TEXT,
            answer_key=COURSE_SETTING_BANK_ANSWER_USER_PROMPT_TEXT,
        )
        group_row: dict[str, Any] = {
            "bank_page_id": page_id,
            "bank_unit_id": bank_unit_id,
            "person_id": pid,
            "course_id": course_id,
            "group_name": (body.group_name or "").strip() or (unit.get("unit_name") or "").strip(),
            "question_system_prompt_text": prompt_texts["question_system_prompt_text"],
            "question_user_prompt_text": prompt_texts["question_user_prompt_text"],
            "qa_count": body.qa_count,
            "question_llm_model": (body.question_llm_model or "").strip(),
            "answer_user_prompt_text": prompt_texts["answer_user_prompt_text"],
            "answer_llm_model": (body.answer_llm_model or "").strip(),
            "for_exam": bool(body.for_exam),
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = insert_bank_child_row("Bank_Group", group_row)
        if not ins.data:
            raise HTTPException(status_code=500, detail="寫入 Bank_Group 失敗（無回傳資料）")
        return to_json_safe(ins.data[0])
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pages/{bank_page_id}/units/{bank_unit_id}/groups",
    summary="List Bank Groups",
    operation_id="bank_list_groups",
)
def list_bank_groups(
    _caller_person_id: PersonId,
    course_id: CourseId,
    bank_page_id: str = PathParam(..., description="Bank.bank_page_id"),
    bank_unit_id: int = PathParam(..., gt=0, description="Bank_Unit 主鍵"),
):
    """列出該單元下所有未刪除題組（含各題組的 Bank_QA，依 question_series_index 升序）。"""
    supabase = get_supabase()
    fetch_bank_unit_in_page(
        supabase, bank_page_id=bank_page_id, bank_unit_id=bank_unit_id, course_id=course_id
    )
    groups_by_unit = groups_by_bank_unit_ids([bank_unit_id], course_id=course_id)
    groups = to_json_safe(groups_by_unit.get(bank_unit_id, []))
    return {"groups": groups, "count": len(groups)}


# ---------------------------------------------------------------------------
# 單一題組（以主鍵淺路徑引用）
# ---------------------------------------------------------------------------


@router.get("/groups/{bank_group_id}", summary="Get Bank Group", operation_id="bank_get_group")
def get_bank_group(
    _caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """讀取單一題組（含其 Bank_QA，依 question_series_index 升序）。"""
    supabase = get_supabase()
    group = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols="*")
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 bank_group_id={bank_group_id} 的 Bank_Group，或已刪除")
    group = to_json_safe(group)
    group["qas"] = to_json_safe(_bank_qa_rows_for_group(supabase, bank_group_id, course_id))
    return group


def _get_bank_group_prompt_field(
    bank_group_id: int, course_id: int, field: str
) -> dict[str, Any]:
    supabase = get_supabase()
    cols = f"bank_group_id, {field}"
    group = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols=cols)
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 bank_group_id={bank_group_id} 的 Bank_Group，或已刪除")
    return {"bank_group_id": bank_group_id, field: group.get(field) or ""}


def _put_bank_group_prompt_field(
    *,
    bank_group_id: int,
    course_id: int,
    caller_person_id: str,
    field: str,
    value: str,
) -> dict[str, Any]:
    supabase = get_supabase()
    require_bank_group_owner(
        supabase, bank_group_id, course_id, caller_person_id, cols="bank_group_id, person_id, course_id"
    )
    supabase.table("Bank_Group").update({
        field: value,
        "updated_at": now_taipei_iso(),
    }).eq("bank_group_id", bank_group_id).eq("deleted", False).execute()
    return _get_bank_group_prompt_field(bank_group_id, course_id, field)


@router.get(
    "/groups/{bank_group_id}/question-system-prompt-text",
    summary="Get Bank Group question_system_prompt_text",
    operation_id="bank_get_group_question_system_prompt_text",
)
def get_bank_group_question_system_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """讀取 Bank_Group.question_system_prompt_text。"""
    return BankGroupQuestionSystemPromptTextResponse(**_get_bank_group_prompt_field(
        bank_group_id, course_id, "question_system_prompt_text"
    ))


@router.put(
    "/groups/{bank_group_id}/question-system-prompt-text",
    summary="Update Bank Group question_system_prompt_text",
    operation_id="bank_put_group_question_system_prompt_text",
)
def put_bank_group_question_system_prompt_text(
    body: openapi_body(
        PutBankGroupQuestionSystemPromptTextRequest,
        {"question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """寫入 Bank_Group.question_system_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return BankGroupQuestionSystemPromptTextResponse(**_put_bank_group_prompt_field(
            bank_group_id=bank_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="question_system_prompt_text",
            value=body.question_system_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /bank/groups/{bank_group_id}/question-system-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/groups/{bank_group_id}/question-user-prompt-text",
    summary="Get Bank Group question_user_prompt_text",
    operation_id="bank_get_group_question_user_prompt_text",
)
def get_bank_group_question_user_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """讀取 Bank_Group.question_user_prompt_text。"""
    return BankGroupQuestionUserPromptTextResponse(**_get_bank_group_prompt_field(
        bank_group_id, course_id, "question_user_prompt_text"
    ))


@router.put(
    "/groups/{bank_group_id}/question-user-prompt-text",
    summary="Update Bank Group question_user_prompt_text",
    operation_id="bank_put_group_question_user_prompt_text",
)
def put_bank_group_question_user_prompt_text(
    body: openapi_body(
        PutBankGroupQuestionUserPromptTextRequest,
        {"question_user_prompt_text": "請就課程內容出一道問答題。"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """寫入 Bank_Group.question_user_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return BankGroupQuestionUserPromptTextResponse(**_put_bank_group_prompt_field(
            bank_group_id=bank_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="question_user_prompt_text",
            value=body.question_user_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /bank/groups/{bank_group_id}/question-user-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/groups/{bank_group_id}/answer-user-prompt-text",
    summary="Get Bank Group answer_user_prompt_text",
    operation_id="bank_get_group_answer_user_prompt_text",
)
def get_bank_group_answer_user_prompt_text(
    _caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """讀取 Bank_Group.answer_user_prompt_text。"""
    return BankGroupAnswerUserPromptTextResponse(**_get_bank_group_prompt_field(
        bank_group_id, course_id, "answer_user_prompt_text"
    ))


@router.put(
    "/groups/{bank_group_id}/answer-user-prompt-text",
    summary="Update Bank Group answer_user_prompt_text",
    operation_id="bank_put_group_answer_user_prompt_text",
)
def put_bank_group_answer_user_prompt_text(
    body: openapi_body(
        PutBankGroupAnswerUserPromptTextRequest,
        {"answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """寫入 Bank_Group.answer_user_prompt_text。僅 person_id 一致者可更新。"""
    try:
        return BankGroupAnswerUserPromptTextResponse(**_put_bank_group_prompt_field(
            bank_group_id=bank_group_id,
            course_id=course_id,
            caller_person_id=caller_person_id,
            field="answer_user_prompt_text",
            value=body.answer_user_prompt_text,
        ))
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /bank/groups/{bank_group_id}/answer-user-prompt-text 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/groups/{bank_group_id}", summary="Update Bank Group", operation_id="bank_update_group")
def update_bank_group(
    body: openapi_body(
        UpdateBankGroupRequest,
        {
            "group_name": "新名稱",
            "qa_count": 8,
            "question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。",
            "question_user_prompt_text": "請就課程內容出一道問答題。",
            "question_llm_model": "",
            "answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。",
            "answer_llm_model": "",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """
    更新題組設定（僅更新有傳入的欄位；可更新 group_name、qa_count、question_system_prompt_text、
    question_user_prompt_text、question_llm_model、answer_user_prompt_text、answer_llm_model）。僅 person_id 一致者可更新。
    """
    try:
        supabase = get_supabase()
        require_bank_group_owner(
            supabase, bank_group_id, course_id, caller_person_id, cols="bank_group_id, person_id, course_id"
        )

        update_payload: dict[str, Any] = {}
        if body.group_name is not None:
            update_payload["group_name"] = body.group_name.strip()
        if body.qa_count is not None:
            update_payload["qa_count"] = int(body.qa_count)
        if body.question_system_prompt_text is not None:
            update_payload["question_system_prompt_text"] = body.question_system_prompt_text
        if body.question_user_prompt_text is not None:
            update_payload["question_user_prompt_text"] = body.question_user_prompt_text
        if body.question_llm_model is not None:
            update_payload["question_llm_model"] = body.question_llm_model.strip()
        if body.answer_user_prompt_text is not None:
            update_payload["answer_user_prompt_text"] = body.answer_user_prompt_text
        if body.answer_llm_model is not None:
            update_payload["answer_llm_model"] = body.answer_llm_model.strip()
        if not update_payload:
            raise HTTPException(status_code=400, detail="未提供任何要更新的欄位")
        update_payload["updated_at"] = now_taipei_iso()

        supabase.table("Bank_Group").update(update_payload).eq("bank_group_id", bank_group_id).eq("deleted", False).execute()
        read = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols="*")
        return to_json_safe(read or {})
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PATCH /bank/groups/{bank_group_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/groups/{bank_group_id}/for-exam", summary="Set Bank Group for_exam flag", operation_id="bank_group_for_exam")
def mark_bank_group_for_exam(
    body: openapi_body(BankGroupForExamRequest, {"for_exam": True}),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """更新 Bank_Group.for_exam（true＝測驗用、false＝取消）。僅 person_id 一致者可更新。"""
    try:
        supabase = get_supabase()
        require_bank_group_owner(
            supabase, bank_group_id, course_id, caller_person_id, cols="bank_group_id, person_id, course_id"
        )
        ts = now_taipei_iso()
        supabase.table("Bank_Group").update({"for_exam": body.for_exam, "updated_at": ts}).eq("bank_group_id", bank_group_id).eq("deleted", False).execute()
        read = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols="*")
        return to_json_safe(read or {})
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PUT /bank/groups/{bank_group_id}/for-exam 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/groups/{bank_group_id}", status_code=200, summary="Delete Bank Group", operation_id="bank_group_delete")
def delete_bank_group(
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """軟刪除題組（僅將此題組 deleted=true，不動其 Bank_QA）。僅 person_id 一致者可刪除。"""
    try:
        supabase = get_supabase()
        group = require_bank_group_owner(
            supabase, bank_group_id, course_id, caller_person_id, cols="bank_group_id, person_id, course_id"
        )
        ts = now_taipei_iso()
        supabase.table("Bank_Group").update({"deleted": True, "updated_at": ts}).eq("bank_group_id", bank_group_id).eq("deleted", False).execute()
        return {
            "message": "已將 Bank_Group 標記為刪除",
            "bank_group_id": bank_group_id,
            "person_id": group.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /bank/groups/{bank_group_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 題組內出題（qa 為 group 的子集合）
# ---------------------------------------------------------------------------


@router.post("/groups/{bank_group_id}/qa/llm-generate", summary="Bank LLM Generate Next QA", operation_id="bank_llm_generate_qa")
def bank_llm_generate_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_group_id: int = PathParam(..., gt=0, description="Bank_Group 主鍵"),
):
    """
    在題組內產生**下一題**（LLM，同步）。一律自 Bank_Group 讀取 question_system_prompt_text（連續出題規定）
    與 question_user_prompt_text（出題 user prompt）；同題組既有題目題幹會作為「已出過題目（勿重複）」一併送入。
    已產生題數達 `qa_count` 上限時回 409。出題成功後新增一筆 Bank_QA 並回傳。無 request body。
    """
    return bank_llm_generate_qa_impl(
        bank_group_id=bank_group_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )


@router.post("/qa/{bank_qa_id}/llm-regenerate", summary="Bank LLM Regenerate QA (in place)", operation_id="bank_llm_regenerate_qa")
def bank_llm_regenerate_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_qa_id: int = PathParam(..., gt=0, description="Bank_QA 主鍵"),
):
    """
    **原地重出同一題**（LLM，同步）：只重新產生這一題的 question_* 內容並覆寫回**同一個 bank_qa_id**，
    不刪除、不新增任何 Bank_QA，也不改動 question_series_index。prompt 一律自所屬 Bank_Group 讀取。
    同題組的**其他**題會作為「已出過題目（勿重複）」送入。重出後本題舊作答／批改會清空。不檢查 qa_count 上限。
    無 request body。
    """
    return bank_llm_regenerate_qa_impl(
        bank_qa_id=bank_qa_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )


# ---------------------------------------------------------------------------
# 單題批改（以主鍵淺路徑引用）
# ---------------------------------------------------------------------------


@router.post("/qa/{bank_qa_id}/llm-answer", summary="Bank Answer QA", operation_id="bank_llm_answer_qa")
async def bank_qa_answer(
    background_tasks: BackgroundTasks,
    body: openapi_body(BankQaAnswerRequest, {"answer_content": "學生作答文字"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_qa_id: int = PathParam(..., gt=0, description="Bank_QA 主鍵"),
):
    """
    非同步批改：以路徑 bank_qa_id 指定要批改的題目，使用所屬題組之 answer_user_prompt_text 批改學生作答。
    unit_type 2／3／4 以 transcript 純 LLM 批改；其餘載入該單元 RAG ZIP。
    回傳 202 + job_id；輪詢 GET /bank/qa/answer-result/{job_id}。
    """
    return await enqueue_bank_qa_answer_job(
        background_tasks,
        caller_person_id,
        course_id,
        bank_qa_id=bank_qa_id,
        answer_content=body.answer_content,
    )


@router.get("/qa/answer-result/{job_id}", summary="Get Bank QA Answer Result", operation_id="bank_qa_answer_result")
async def get_bank_qa_answer_result(_person_id: PersonId, course_id: CourseId, job_id: str):
    """輪詢批改結果。status: pending | ready | error；ready 時 result 含 quiz_comments、bank_qa_id，並回讀整列 Bank_QA。"""
    if job_id not in _bank_answer_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出批改）",
            },
        )
    data = _bank_answer_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "llm_error": data.get("llm_error"),
    }
    bank_qa_row: dict[str, Any] | None = None
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            rid = res.get("bank_qa_id")
            if rid is not None:
                try:
                    rid_int = int(rid)
                except (TypeError, ValueError):
                    rid_int = 0
                if rid_int > 0:
                    try:
                        supabase = get_supabase()

                        def build_sel(with_course_filter: bool):
                            q = (
                                supabase.table("Bank_QA")
                                .select("*")
                                .eq("bank_qa_id", rid_int)
                                .eq("deleted", False)
                            )
                            if with_course_filter and course_id is not None:
                                q = q.eq("course_id", course_id)
                            return q.limit(1)

                        q = execute_with_course_id_fallback("Bank_QA", build_sel, course_id)
                        if q.data:
                            bank_qa_row = to_json_safe(q.data[0])
                    except Exception as e:
                        _logger.warning("answer-result 讀取 Bank_QA 失敗 job_id=%s: %s", job_id, e)
        out["bank_qa"] = bank_qa_row
    return out


@router.delete("/qa/{bank_qa_id}", status_code=200, summary="Delete Bank QA", operation_id="bank_qa_delete")
def delete_bank_qa(
    caller_person_id: PersonId,
    course_id: CourseId,
    bank_qa_id: int = PathParam(..., gt=0, description="Bank_QA 主鍵"),
):
    """軟刪除單一題目（Bank_QA.deleted=true）。僅 person_id 一致者可刪除。"""
    try:
        supabase = get_supabase()
        qa = _fetch_bank_qa_row(
            supabase, bank_qa_id, course_id, cols="bank_qa_id, bank_group_id, person_id, course_id"
        )
        if not qa:
            raise HTTPException(status_code=404, detail=f"找不到 bank_qa_id={bank_qa_id} 的 Bank_QA，或已刪除")
        if (qa.get("person_id") or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Bank_QA")
        ts = now_taipei_iso()
        supabase.table("Bank_QA").update({"deleted": True, "updated_at": ts}).eq("bank_qa_id", bank_qa_id).eq("deleted", False).execute()
        return {
            "message": "已將 Bank_QA 標記為刪除",
            "bank_qa_id": bank_qa_id,
            "bank_group_id": qa.get("bank_group_id"),
            "person_id": qa.get("person_id"),
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /bank/qa/{bank_qa_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))
