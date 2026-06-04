"""
個人分析 API 模組（資料皆在 Person_Analysis，不再使用 Course_Setting／System_Setting）。

- GET /person-analysis/analysis：query person_id、course_id，回傳最新一筆。
- GET /person-analysis/llm-analysis：依作答產生弱點報告並寫入 Person_Analysis。
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from services.exam_queries import (
    exams_by_page_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    exam_tab_quizzes_response,
    quizzes_by_person_id,
)
from services.person_analysis_setting import (
    COURSE_WIDE_PERSON_ANALYSIS_PERSON_ID,
    fetch_latest_person_analysis_result_row,
    fetch_person_analysis_instruction_text,
    fetch_person_analysis_user_prompt_for_llm,
    resolve_login_person_id,
    save_person_analysis_setting,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.llm_key import get_person_analysis_api_key, get_rag_llm_model
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
        default=None, description="Person_Analysis 主鍵"
    )
    person_id: Optional[str] = Field(default=None, description="登入帳號；課程共用為空字串")
    course_id: Optional[int] = None
    analysis_user_prompt_text: Optional[str] = Field(
        default=None,
        description="教師分析指令（純文字；僅來自 analysis_text 為空的指令列）",
    )
    analysis_prompt_text: Optional[str] = Field(
        default=None,
        description="教師分析指令（純文字；與 analysis_user_prompt_text 同源，僅來自 API 寫入之指令列）",
    )
    analysis_text: Optional[str] = Field(
        default=None, description="已儲存的弱點報告 Markdown 全文",
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


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
    `analysis_user_prompt_text`／`analysis_prompt_text` 僅來自教師指令列（API 寫入）；`analysis_text` 來自最新 LLM 結果列。
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
            analysis_prompt_text=instr_text or None,
            analysis_text=safe_result.get("analysis_text"),
            created_at=safe_result.get("created_at"),
            updated_at=safe_result.get("updated_at"),
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
    教師指令自 Person_Analysis 讀取（該生優先，其次課程共用 person_id 空字串）。
    成功後將報告寫入 Person_Analysis（結果列不寫入 analysis_prompt_text）。
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
            weakness_report, _ = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_PERSON,
                llm_model=analysis_llm_model,
            )
            if weakness_report:
                saved = save_person_analysis_setting(
                    login_pid,
                    course_id,
                    weakness_report,
                )
                if not saved:
                    logger.error(
                        "person_llm_analysis: LLM ok but Person_Analysis insert failed "
                        "person_id=%s course_id=%s",
                        login_pid,
                        course_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"弱點報告已產生但寫入 Person_Analysis 失敗 "
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
