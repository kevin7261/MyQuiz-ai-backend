"""
課程分析 API 模組。
依 course_id 查詢 Exam_Quiz 資料。新 schema 中答案欄位（answer_content／answer_critique）
直接內嵌於 Exam_Quiz，不再有獨立的 Exam_Answer 表。
- GET /course-analysis/quizzes：依 course_id 取得已作答的 Exam_Quiz（answer_content 非空），
  依 exam_tab_id 分群對應 Exam；每筆 Exam 的題目結構與 GET /exam/tabs 相同（quizzes[]，Exam_Quiz 含 follow_up 鏈；作答內嵌於各題列）。
  另帶 weakness_report：每次請求皆呼叫 LLM 產生弱點報告（有 LLM API Key 且成功呼叫時為模型回覆原文，否則 null）。

**課程分析 user prompt** 取自 `System_Setting.key=course_analysis_user_prompt_text`（與 GET/PUT `/system-settings/course_analysis_user_prompt_text` 同源）。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from routers.system_settings import (
    SYSTEM_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY,
    fetch_system_setting_text,
)
from services.exam_queries import (
    exams_by_tab_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    exam_tab_quizzes_response,
    quizzes_by_course_id,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.serialization import to_json_safe
from utils.llm_key import get_llm_api_key

router = APIRouter(prefix="/course-analysis", tags=["course analysis"])

ANALYSIS_LABEL_COURSE = "課程分析"


class ListQuizzesResponse(BaseModel):
    """GET /course-analysis/quizzes 回應。exams[] 每筆與 GET /exam/tabs 相同含 quizzes[]；weakness_report 為 LLM 弱點報告（失敗時為 null）。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：LLM `message.content` 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )


@router.get("/quizzes", response_model=ListQuizzesResponse)
def list_exam_quizzes(_person_id: PersonId, course_id: CourseId):
    """
    依 course_id 取得已作答的 Exam_Quiz（answer_content 非空），依 exam_tab_id 分群；
    每筆 Exam 的 quizzes 形狀與 GET /exam/tabs 一致。
    weakness_report：每次請求皆嘗試呼叫 LLM 產生；弱點報告 user 訊息會併入 System_Setting
    `course_analysis_user_prompt_text`（與 `/system-settings/course_analysis_user_prompt_text` 同源）。
    必填 query course_id。
    """
    try:
        quizzes = quizzes_by_course_id(course_id)
        quizzes_with_answers = [q for q in quizzes if quiz_has_answer(q)]

        tab_ids: list[str] = list(dict.fromkeys(
            str(q.get("exam_tab_id")) for q in quizzes_with_answers if q.get("exam_tab_id") is not None
        ))

        exam_rows = exams_by_tab_ids(tab_ids)
        quizzes_by_tab: dict[str, list[dict]] = {tid: [] for tid in tab_ids}
        for q in quizzes_with_answers:
            tid = q.get("exam_tab_id")
            if tid is not None:
                quizzes_by_tab.setdefault(str(tid), []).append(q)

        flat_for_enrich = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_for_enrich)
        ensure_exam_quiz_rag_id_keys(flat_for_enrich)

        for row in exam_rows:
            tid = str(row.get("exam_tab_id") or "")
            row["quizzes"] = exam_tab_quizzes_response(quizzes_by_tab.get(tid, []))

        data = to_json_safe(exam_rows)
        weakness_report: Optional[str] = None
        api_key = get_llm_api_key()
        if api_key:
            setting_prompt = fetch_system_setting_text(
                SYSTEM_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
            )
            weakness_report = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_COURSE,
            )
        return ListQuizzesResponse(exams=data, count=len(data), weakness_report=weakness_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
