"""
課程分析 API 模組（資料存於 Course_Analysis；見 services.analysis_setting）。

person_id 一律為呼叫 API 的登入帳號。
- analysis_prompt_text ↔ answer_user_prompt_text（PUT /rag/course_analysis_user_prompt_text）
- analysis_text ↔ answer_critique（POST /course-analysis/llm-analysis）
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from services.analysis_setting import (
    fetch_course_analysis_stored,
    fetch_course_analysis_user_prompt_for_llm,
    resolve_login_person_id,
    save_course_analysis_setting,
)
from services.exam_queries import (
    exams_by_page_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    exam_tab_quizzes_response,
    quizzes_by_course_id,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.llm_key import get_course_analysis_api_key, get_rag_llm_model
from utils.serialization import to_json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/course-analysis", tags=["course analysis"])

ANALYSIS_LABEL_COURSE = "課程分析"


def _caller_person_id_or_404(person_id: str) -> str:
    login = resolve_login_person_id(person_id)
    if not login:
        raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
    return login


class CourseStoredAnalysisResponse(BaseModel):
    """GET /course-analysis/analysis 回應；無紀錄時各欄位為 null。"""
    course_analysis_id: Optional[int] = Field(
        default=None, description="Course_Analysis 主鍵"
    )
    person_id: Optional[str] = Field(default=None, description="呼叫者登入帳號")
    course_id: Optional[int] = None
    analysis_user_prompt_text: Optional[str] = Field(
        default=None,
        description="教師分析指令（對應 answer_user_prompt_text）",
    )
    analysis_prompt_text: Optional[str] = Field(
        default=None,
        description="與 analysis_user_prompt_text 同源（DB 欄位 analysis_prompt_text）",
    )
    analysis_text: Optional[str] = Field(
        default=None, description="弱點報告 Markdown（對應 answer_critique）",
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CourseLlmAnalysisResponse(BaseModel):
    """POST /course-analysis/llm-analysis 回應。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：LLM message.content 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )
    llm_error: Optional[str] = Field(
        default=None,
        description="LLM 呼叫失敗或未設定 API Key 時的錯誤原因；成功時為 null",
    )
    analysis_llm_model: str = Field(
        ...,
        description="本次弱點分析實際使用的 LLM 模型（Course_Setting key=llm-model）。API Key 為 rag-api-key",
    )


def _stored_to_response(stored: Optional[dict]) -> CourseStoredAnalysisResponse:
    if not stored:
        return CourseStoredAnalysisResponse()
    safe = to_json_safe(stored)
    prompt = safe.get("analysis_prompt_text")
    return CourseStoredAnalysisResponse(
        course_analysis_id=safe.get("course_analysis_id"),
        person_id=safe.get("person_id"),
        course_id=safe.get("course_id"),
        analysis_user_prompt_text=prompt,
        analysis_prompt_text=prompt,
        analysis_text=safe.get("analysis_text"),
        created_at=safe.get("created_at"),
        updated_at=safe.get("updated_at"),
    )


@router.get("/analysis", response_model=CourseStoredAnalysisResponse)
def get_course_stored_analysis(person_id: PersonId, course_id: CourseId):
    """
    取值：不呼叫 LLM。必填 query `person_id`（呼叫者）、`course_id`。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        stored = fetch_course_analysis_stored(caller, course_id)
        return _stored_to_response(stored)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/llm-analysis", response_model=CourseLlmAnalysisResponse)
def course_llm_analysis(person_id: PersonId, course_id: CourseId):
    """
    必填 query `person_id`（呼叫者）、`course_id`。
    成功後 UPDATE 呼叫者 Course_Analysis 列 analysis_text。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        quizzes = quizzes_by_course_id(course_id)
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
        llm_error: Optional[str] = None
        if not quizzes_with_answers:
            llm_error = "無已作答或已評級題目，無法產生弱點報告（未寫入 Course_Analysis）"
        api_key = get_course_analysis_api_key(course_id)
        if not llm_error and not api_key:
            llm_error = "未設定 API Key：PUT /rag/llm_api_key（Course_Setting key=rag-api-key，依 course_id）"
        elif not llm_error:
            setting_prompt = fetch_course_analysis_user_prompt_for_llm(caller, course_id)
            weakness_report, _, llm_err = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_COURSE,
                llm_model=analysis_llm_model,
            )
            if llm_err:
                llm_error = llm_err
            if weakness_report:
                saved = save_course_analysis_setting(caller, course_id, weakness_report)
                if not saved:
                    logger.error(
                        "course_llm_analysis: LLM ok but Course_Analysis update failed "
                        "person_id=%s course_id=%s",
                        caller,
                        course_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"弱點報告已產生但寫入 Course_Analysis 失敗 "
                            f"(person_id={caller}, course_id={course_id})"
                        ),
                    )
        return CourseLlmAnalysisResponse(
            exams=data,
            count=len(data),
            weakness_report=weakness_report,
            llm_error=llm_error,
            analysis_llm_model=analysis_llm_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
