"""
分析 API：依 person_id 查詢 Exam_Quiz / Exam_Answer 等分析用資料。
- GET /analysis/quizzes-by-person/{person_id}：依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆帶關聯的 Exam_Answer。
  回傳格式與 GET /rag/rags、GET /exam/exams 的題目答案內容一致（每筆 quiz 含 quiz_content、quiz_hint、reference_answer、quiz_metadata，answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
  可選參數 language、llm_api_key：若提供 llm_api_key，會依題目／參考答案／使用者答案／答案分析結果彙整弱點，由 AI 產生「全部弱點分析」報告（Markdown），放在 weakness_report 欄位。
"""

import json
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Path as PathParam
from openai import OpenAI
from pydantic import BaseModel, Field

from routers.exam import _answers_by_exam_quiz_ids, _quizzes_by_person_id
from utils.json_utils import to_json_safe

router = APIRouter(prefix="/analysis", tags=["analysis"])


class ListQuizzesByPersonResponse(BaseModel):
    """GET /analysis/quizzes-by-person/{person_id} 回應：格式同 rag/exam 的題目答案，每筆 quiz 帶 answers；可選帶全部弱點分析。"""
    quizzes: list[dict]
    count: int
    weakness_report: Optional[str] = Field(default=None, description="依題目／參考答案／使用者答案／答案分析結果彙整後由 AI 產生的 Markdown 弱點報告；未傳 llm_api_key 時為 None")


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


def _generate_weakness_report_md(quizzes: list[dict], lang: Literal["en", "zh"], api_key: str) -> Optional[str]:
    """依題目／參考答案／使用者答案／答案分析結果彙整弱點，呼叫 LLM 產生 Markdown 報告。無弱點或 API 失敗時回傳 None。"""
    all_weaknesses = _collect_weaknesses_from_quizzes(quizzes)
    if not all_weaknesses:
        return None
    weakness_text = "\n".join(all_weaknesses[:60])
    if lang == "en":
        prompt = f"Analyze the following learning weaknesses from quiz feedback and produce a clear, actionable Markdown report.\n\nWeaknesses:\n{weakness_text}\n\nProduce Markdown report only."
    else:
        prompt = f"""你是教學顧問。請根據以下來自測驗回饋的學習弱點，製作一份 Markdown 報告。
弱點列表：
{weakness_text}

請製作 Markdown 報告。
**請務必使用繁體中文 (Traditional Chinese) 撰寫所有報告內容。**
"""
    client = OpenAI(api_key=api_key)
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception:
        return None
    return (r.choices[0].message.content or "").strip() or None


@router.get("/quizzes-by-person/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    person_id: str = PathParam(..., description="要查詢的 person_id"),
    language: Literal["en", "zh"] = "zh",
    llm_api_key: Optional[str] = None,
):
    """
    依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆 quiz 帶關聯的 Exam_Answer（answers）。
    回傳題目／答案的 JSON 結構與 GET /rag/rags、GET /exam/exams 一致（quiz_content、quiz_hint、reference_answer、quiz_metadata；answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
    若提供 llm_api_key，會依題目／參考答案／使用者答案／答案分析結果彙整弱點並由 AI 產生全部弱點分析報告，放在 weakness_report。
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
        data = to_json_safe(quizzes)
        weakness_report: Optional[str] = None
        if (llm_api_key or "").strip():
            weakness_report = _generate_weakness_report_md(data, language, (llm_api_key or "").strip())
        return ListQuizzesByPersonResponse(quizzes=data, count=len(data), weakness_report=weakness_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
