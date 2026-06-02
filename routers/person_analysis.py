"""
個人分析 API 模組。
依 person_id、course_id 查詢 Exam_Quiz 資料。新 schema 答案欄位直接內嵌於 Exam_Quiz（answer_content, answer_critique），不再有獨立的 Exam_Answer 表。
- GET /person-analysis/quizzes/{person_id}：依 person_id、course_id 取得已作答的 Exam_Quiz（answer_content 非空），
  依 exam_page_id 分群回傳 Exam；每筆 Exam 的題目結構與 GET /exam/tabs 相同（quizzes[]，Exam_Quiz 含 follow_up 鏈，含 enrich／rag 鍵）。
  另帶 weakness_report：每次請求皆呼叫 LLM 產生弱點報告（有 LLM API Key 且成功呼叫時為模型回覆原文，否則 null）。

重要：弱點報告與出題／批改相同，系統與使用者訊息皆為 **Markdown**；本路由**不**使用 `response_format=json_object`。
**個人分析 user prompt** 取自 `Course_Setting.key=person_analysis_user_prompt_text`（與 GET/PUT `/rag/person_analysis_user_prompt_text` 同源），嵌入 user 訊息之 **`## 個人分析 user prompt`**。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Path as PathParam

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from utils.course_setting import (
    COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY,
    fetch_course_setting_text,
)

# 向後相容別名
SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY = (
    COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY
)
fetch_system_setting_text = fetch_course_setting_text
from services.exam_queries import (
    exams_by_page_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    exam_tab_quizzes_response,
    quizzes_by_person_id,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.serialization import to_json_safe
from utils.llm_key import get_exam_api_key

router = APIRouter(prefix="/person-analysis", tags=["person analysis"])

ANALYSIS_LABEL_PERSON = "個人分析"


class ListQuizzesByPersonResponse(BaseModel):
    """GET /person-analysis/quizzes/{person_id} 回應。exams[] 每筆與 GET /exam/tabs 相同含 quizzes[]；weakness_report 為 LLM 弱點報告（失敗時為 null）。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：LLM `message.content` 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )


@router.get("/quizzes/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    caller_person_id: PersonId,
    course_id: CourseId,
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id、course_id 取得已作答的 Exam_Quiz（answer_content 非空），依 exam_page_id 分群後對應 Exam；
    每筆 Exam 的 quizzes 形狀與 GET /exam/tabs 一致（題目為完整 Exam_Quiz 列，含作答欄位）。
    weakness_report：每次請求皆嘗試呼叫 LLM 產生；成功時為 `message.content` 原文，否則為 null。
    弱點報告 user 訊息會併入 Course_Setting `person_analysis_user_prompt_text`
    （與 `/rag/person_analysis_user_prompt_text` 同源）。
    query 的 person_id 須與路徑 {person_id} 一致；必填 query course_id。
    """
    try:
        path_pid = (person_id or "").strip()
        if path_pid != caller_person_id:
            raise HTTPException(status_code=403, detail="路徑 person_id 與 query 不一致")

        quizzes = quizzes_by_person_id(path_pid, course_id=course_id)
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
        weakness_report: Optional[str] = None
        api_key = get_exam_api_key(course_id)
        if api_key:
            setting_prompt = fetch_system_setting_text(
                SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
            )
            weakness_report = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_PERSON,
            )
        return ListQuizzesByPersonResponse(exams=data, count=len(data), weakness_report=weakness_report)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
