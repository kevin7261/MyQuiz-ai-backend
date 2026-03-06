"""
分析 API：依 person_id 查詢 Exam_Quiz / Exam_Answer 等分析用資料。
- GET /analysis/quizzes-by-person/{person_id}：依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆帶關聯的 Exam_Answer。
  回傳格式與 GET /rag/rags、GET /exam/exams 的題目答案內容一致（每筆 quiz 含 quiz_content、quiz_hint、reference_answer、quiz_metadata，answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
- POST /analysis/weakness-report：輸入為 quizzes-by-person 的輸出 JSON，彙整各題 answer_feedback_metadata 中的 weaknesses，由 AI 產生學習弱點報告（Markdown）。
"""

import json
from datetime import date, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Path as PathParam
from openai import OpenAI
from pydantic import BaseModel, Field

from routers.exam import _answers_by_exam_quiz_ids, _quizzes_by_person_id

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _to_json_safe(obj: Any) -> Any:
    """將 Supabase/DB 回傳值轉成可 JSON 序列化的型別（與 rag、exam 一致，避免 500）。"""
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "keys") and not isinstance(obj, dict):
        return _to_json_safe(dict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return obj


class ListQuizzesByPersonResponse(BaseModel):
    """GET /analysis/quizzes-by-person/{person_id} 回應：格式同 rag/exam 的題目答案，每筆 quiz 帶 answers。"""
    quizzes: list[dict]
    count: int


class WeaknessReportRequest(BaseModel):
    """POST /analysis/weakness-report 請求：格式同 GET /analysis/quizzes-by-person 的輸出。"""
    quizzes: list[dict] = Field(..., description="題目與答案列表，每筆含 answers（含 answer_feedback_metadata）")
    language: Literal["en", "zh"] = Field(default="zh", description="報告語言：en=英文，zh=繁體中文")
    llm_api_key: str = Field(..., description="LLM API key，用於產生報告")


class WeaknessReportResponse(BaseModel):
    """POST /analysis/weakness-report 回應。"""
    report: str = Field(..., description="Markdown 格式的學習弱點報告")
    weaknesses_count: int = Field(..., description="彙整的弱點條數")


def _collect_weaknesses_from_quizzes(quizzes: list[dict]) -> list[str]:
    """從 quizzes-by-person 格式的 quizzes 中，收集所有 answer_feedback_metadata 的 weaknesses。"""
    all_weaknesses: list[str] = []
    for quiz in quizzes or []:
        answers = quiz.get("answers") or []
        for ans in answers:
            meta = ans.get("answer_feedback_metadata")
            if not meta:
                continue
            if isinstance(meta, dict):
                data = meta
            elif isinstance(meta, str):
                try:
                    data = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                continue
            if isinstance(data.get("weaknesses"), list):
                all_weaknesses.extend(w for w in data["weaknesses"] if isinstance(w, str) and w.strip())
    return all_weaknesses


@router.get("/quizzes-by-person/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆 quiz 帶關聯的 Exam_Answer（answers）。
    回傳題目／答案的 JSON 結構與 GET /rag/rags、GET /exam/exams 一致（quiz_content、quiz_hint、reference_answer、quiz_metadata；answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
    """
    try:
        quizzes = _quizzes_by_person_id(person_id)
        quiz_ids = []
        for row in quizzes:
            qid = row.get("exam_quiz_id")
            if qid is not None:
                try:
                    quiz_ids.append(int(qid))
                except (TypeError, ValueError):
                    pass
        quiz_ids = list(dict.fromkeys(quiz_ids))
        answers_by_quiz = _answers_by_exam_quiz_ids(quiz_ids)
        for quiz in quizzes:
            qid = quiz.get("exam_quiz_id")
            qid_int = int(qid) if qid is not None else None
            quiz["answers"] = answers_by_quiz.get(qid_int, []) if qid_int is not None else []
        # 與 rag、exam 一致：轉成可 JSON 序列化（datetime 等轉成 ISO 字串）
        data = _to_json_safe(quizzes)
        return ListQuizzesByPersonResponse(quizzes=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weakness-report", response_model=WeaknessReportResponse)
def generate_weakness_report(body: WeaknessReportRequest):
    """
    依 quizzes-by-person 的輸出 JSON，彙整各題答案的 feedback 弱點，由 AI 產生學習弱點報告（Markdown）。
    輸入格式同 GET /analysis/quizzes-by-person/{person_id} 的回應（quizzes 含題目、答案、分數、分析結果；hint 不使用）。
    """
    all_weaknesses = _collect_weaknesses_from_quizzes(body.quizzes)
    if not all_weaknesses:
        raise HTTPException(
            status_code=422,
            detail="No sufficient data or no weaknesses recorded in answer_feedback_metadata.",
        )
    weakness_text = "\n".join(all_weaknesses[:60])
    lang = body.language
    if lang == "en":
        prompt = f"Analyze the following learning weaknesses from quiz feedback and produce a clear, actionable Markdown report.\n\nWeaknesses:\n{weakness_text}\n\nProduce Markdown report only."
    else:
        prompt = f"""你是教學顧問。請根據以下來自測驗回饋的學習弱點，製作一份 Markdown 報告。
弱點列表：
{weakness_text}

請製作 Markdown 報告。
**請務必使用繁體中文 (Traditional Chinese) 撰寫所有報告內容。**
"""
    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 llm_api_key")
    client = OpenAI(api_key=api_key)
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e!s}")
    report = (r.choices[0].message.content or "").strip()
    return WeaknessReportResponse(report=report, weaknesses_count=len(all_weaknesses))
