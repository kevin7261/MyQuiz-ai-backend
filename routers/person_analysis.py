"""
個人分析 API 模組（資料皆在 Person_Analysis_Setting，不再使用 Course_Setting／System_Setting）。

- GET /person-analysis/analysis：query person_id、course_id，回傳最新一筆。
- PUT /person-analysis/analysis：寫入教師分析指令（analysis_prompt_text 純文字）。
- GET /person-analysis/llm-analysis：依作答產生弱點報告並寫入 Person_Analysis_Setting。
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from routers.course_settings import _require_developer_or_manager_for_analysis_prompt_write
from services.exam_queries import (
    exams_by_page_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    exam_tab_quizzes_response,
    quizzes_by_person_id,
)
from services.person_analysis_setting import (
    COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID,
    fetch_latest_person_analysis_instruction_row,
    fetch_latest_person_analysis_result_row,
    fetch_person_analysis_instruction_text,
    fetch_person_analysis_user_prompt_for_llm,
    resolve_login_person_id,
    save_person_analysis_prompt_instruction,
    save_person_analysis_setting,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.llm_key import get_person_analysis_api_key, get_rag_llm_model
from utils.openapi import openapi_body
from utils.serialization import to_json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/person-analysis", tags=["person analysis"])

ANALYSIS_LABEL_PERSON = "個人分析"


def require_target_person_id(
    person_id_q: Optional[str] = Query(
        None,
        alias="person_id",
        description="學生登入帳號；課程共用指令請傳空字串",
    ),
) -> str:
    if person_id_q is None:
        raise HTTPException(status_code=400, detail="請傳入 query 參數 person_id")
    return str(person_id_q).strip()


TargetPersonId = Annotated[str, Depends(require_target_person_id)]


def _login_person_id_or_404(person_id: str) -> str:
    if person_id == COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID:
        return COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID
    login = resolve_login_person_id(person_id)
    if not login:
        raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
    return login


class PersonStoredAnalysisResponse(BaseModel):
    """GET /person-analysis/analysis 回應；無紀錄時各欄位為 null。"""
    person_analysis_id: Optional[int] = Field(
        default=None, description="Person_Analysis_Setting 主鍵"
    )
    person_id: Optional[str] = Field(default=None, description="登入帳號；課程共用為空字串")
    course_id: Optional[int] = None
    analysis_user_prompt_text: Optional[str] = Field(
        default=None,
        description="教師分析指令（純文字；僅來自 analysis_text 為空的指令列）",
    )
    analysis_prompt_text: Optional[str] = Field(
        default=None,
        description="LLM 結果列：送交 LLM 的 prompt JSON；指令列時同 analysis_user_prompt_text",
    )
    analysis_text: Optional[str] = Field(
        default=None, description="已儲存的弱點報告 Markdown 全文",
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PutPersonAnalysisPromptRequest(BaseModel):
    """PUT /person-analysis/analysis 的 body。"""

    analysis_prompt_text: str = Field(..., description="個人分析使用者 Prompt 文字（純文字）")


class PersonLlmAnalysisResponse(BaseModel):
    """GET /person-analysis/llm-analysis 回應。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：LLM `message.content` 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )
    analysis_llm_model: str = Field(
        ...,
        description="本次弱點分析實際使用的 LLM 模型（Course_Setting key=llm-model）。API Key 為 exam-api-key",
    )


@router.get("/analysis", response_model=PersonStoredAnalysisResponse)
def get_person_stored_analysis(
    person_id: TargetPersonId,
    course_id: CourseId,
):
    """
    取值：不呼叫 LLM。必填 query `person_id`、`course_id`。
    `analysis_user_prompt_text` 僅來自教師指令列；`analysis_text`／`analysis_prompt_text`（JSON）來自最新 LLM 結果列。
    DB 無任何列時各欄位為 null（不會回傳程式內建 prompt 模板）。
    """
    try:
        db_pid = _login_person_id_or_404(person_id)
        instr_id, instr_text = fetch_person_analysis_instruction_text(db_pid, course_id)
        result_row = fetch_latest_person_analysis_result_row(db_pid, course_id)
        if not instr_text and not result_row:
            return PersonStoredAnalysisResponse()
        safe_result = to_json_safe(result_row) if result_row else {}
        return PersonStoredAnalysisResponse(
            person_analysis_id=safe_result.get("person_analysis_id") or instr_id,
            person_id=safe_result.get("person_id") or db_pid,
            course_id=safe_result.get("course_id") or course_id,
            analysis_user_prompt_text=instr_text or None,
            analysis_prompt_text=safe_result.get("analysis_prompt_text")
            or (instr_text if instr_text else None),
            analysis_text=safe_result.get("analysis_text"),
            created_at=safe_result.get("created_at"),
            updated_at=safe_result.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/analysis", response_model=PersonStoredAnalysisResponse)
def put_person_analysis_prompt(
    body: openapi_body(
        PutPersonAnalysisPromptRequest,
        {"analysis_prompt_text": "string"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    person_id: TargetPersonId,
):
    """
    寫入教師分析指令至 Person_Analysis_Setting（純文字，不呼叫 LLM）。
    query `person_id` 空字串為課程共用；否則為該生登入帳號。僅開發者／管理者可寫入。
    """
    _require_developer_or_manager_for_analysis_prompt_write(caller_person_id, course_id)
    try:
        db_pid = _login_person_id_or_404(person_id)
        row = save_person_analysis_prompt_instruction(
            db_pid, course_id, body.analysis_prompt_text
        )
        if not row:
            raise HTTPException(
                status_code=500,
                detail=(
                    "寫入 Person_Analysis_Setting 失敗；"
                    "請確認 Supabase 已將 person_id 改為 varchar(255)，"
                    "或查看伺服器 log（legacy bigint 會自動改用 user_id 寫入）"
                ),
            )
        safe = to_json_safe(row)
        prompt_text = (safe.get("analysis_prompt_text") or "").strip() or None
        return PersonStoredAnalysisResponse(
            person_analysis_id=safe.get("person_analysis_id"),
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            analysis_user_prompt_text=prompt_text,
            analysis_prompt_text=prompt_text,
            analysis_text=safe.get("analysis_text"),
            created_at=safe.get("created_at"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/llm-analysis", response_model=PersonLlmAnalysisResponse)
def person_llm_analysis(
    person_id: TargetPersonId,
    course_id: CourseId,
):
    """
    必填 query `person_id`（學生登入帳號）、`course_id`。
    教師指令自 Person_Analysis_Setting 讀取（該生優先，其次課程共用 person_id 空字串）。
    成功後將 prompt JSON 與報告寫入 Person_Analysis_Setting。
    """
    try:
        login_pid = _login_person_id_or_404(person_id)
        quizzes = quizzes_by_person_id(login_pid, course_id=course_id)
        quizzes_with_answers = [q for q in quizzes if quiz_has_answer(q)]

        page_ids: list[str] = list(dict.fromkeys(
            str(q.get("exam_page_id")) for q in quizzes_with_answers if q.get("exam_page_id") is not None
        ))
        exam_rows = exams_by_page_ids(page_ids)
        quizzes_by_tab: dict[str, list[dict]] = {tid: [] for tid in page_ids}
        for q in quizzes_with_answers:
            tid = q.get("exam_page_id")
            if tid is not None:
                quizzes_by_tab.setdefault(str(tid), []).append(q)

        flat_for_enrich = [qz for tid in page_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_for_enrich)
        ensure_exam_quiz_rag_id_keys(flat_for_enrich)

        for row in exam_rows:
            tid = str(row.get("exam_page_id") or "")
            row["quizzes"] = exam_tab_quizzes_response(quizzes_by_tab.get(tid, []))

        data = to_json_safe(exam_rows)
        analysis_llm_model = get_rag_llm_model(course_id)
        weakness_report: Optional[str] = None
        api_key = get_person_analysis_api_key(course_id)
        if api_key:
            setting_prompt = fetch_person_analysis_user_prompt_for_llm(login_pid, course_id)
            weakness_report, analysis_prompt_text = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_PERSON,
                llm_model=analysis_llm_model,
            )
            if weakness_report and analysis_prompt_text:
                saved = save_person_analysis_setting(
                    login_pid,
                    course_id,
                    analysis_prompt_text,
                    weakness_report,
                )
                if not saved:
                    logger.error(
                        "person_llm_analysis: LLM ok but Person_Analysis_Setting insert failed "
                        "person_id=%s course_id=%s",
                        login_pid,
                        course_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"弱點報告已產生但寫入 Person_Analysis_Setting 失敗 "
                            f"(person_id={login_pid}, course_id={course_id})"
                        ),
                    )
        return PersonLlmAnalysisResponse(
            exams=data,
            count=len(data),
            weakness_report=weakness_report,
            analysis_llm_model=analysis_llm_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
