"""
課程分析 API 模組（指令／報告存於 Course_Setting；key 見 services.analysis_setting）。

- GET /course-analysis/analysis：query course_id，回傳最新一筆。
- GET /course-analysis/llm-analysis：依全課程作答產生弱點報告並寫入 Course_Setting。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from services.analysis_setting import (
    fetch_course_analysis_instruction_text,
    fetch_course_analysis_user_prompt_for_llm,
    fetch_latest_course_analysis_result_row,
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
from utils.llm_key import get_rag_llm_model, get_weakness_analysis_api_key
from utils.serialization import to_json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/course-analysis", tags=["course analysis"])

ANALYSIS_LABEL_COURSE = "課程分析"


class CourseStoredAnalysisResponse(BaseModel):
    """GET /course-analysis/analysis 回應；無紀錄時各欄位為 null。"""
    course_analysis_id: Optional[int] = Field(
        default=None, description="課程分析設定識別（Course_Setting.course_setting_id）"
    )
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


class CourseLlmAnalysisResponse(BaseModel):
    """GET /course-analysis/llm-analysis 回應。"""
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


@router.get("/analysis", response_model=CourseStoredAnalysisResponse)
def get_course_stored_analysis(course_id: CourseId):
    """
    取值：不呼叫 LLM。必填 query `course_id`。
    `analysis_user_prompt_text`／`analysis_prompt_text` 僅來自教師指令列（API 寫入）；`analysis_text` 來自最新 LLM 結果列。
    DB 無任何列時各欄位為 null。
    """
    try:
        instr_id, instr_text = fetch_course_analysis_instruction_text(course_id)
        result_row = fetch_latest_course_analysis_result_row(course_id)
        if not instr_text and not result_row:
            return CourseStoredAnalysisResponse()
        safe_result = to_json_safe(result_row) if result_row else {}
        return CourseStoredAnalysisResponse(
            course_analysis_id=safe_result.get("course_analysis_id") or instr_id,
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


@router.get("/llm-analysis", response_model=CourseLlmAnalysisResponse)
def course_llm_analysis(_person_id: PersonId, course_id: CourseId):
    """
    必填 query `course_id`。
    教師指令自 Course_Setting 讀取；成功後將報告寫入 Course_Setting（key=course_analysis_text）。
    """
    try:
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
        api_key = get_weakness_analysis_api_key(course_id)
        if api_key:
            setting_prompt = fetch_course_analysis_user_prompt_for_llm(course_id)
            weakness_report, _ = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_COURSE,
                llm_model=analysis_llm_model,
            )
            if weakness_report:
                saved = save_course_analysis_setting(course_id, weakness_report)
                if not saved:
                    logger.error(
                        "course_llm_analysis: LLM ok but Course_Setting save failed "
                        "course_id=%s",
                        course_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"弱點報告已產生但寫入 Course_Setting 失敗 (course_id={course_id})",
                    )
        return CourseLlmAnalysisResponse(
            exams=data,
            count=len(data),
            weakness_report=weakness_report,
            analysis_llm_model=analysis_llm_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
