"""
個人分析 API 模組。
依 person_id 查詢 Exam_Quiz / Exam_Answer 等分析用資料。
- GET /person-analysis/quizzes/{person_id}：依 person_id 取得該使用者在 Exam_Quiz 的資料，**僅回傳在 Exam_Answer 有對應答案的 quiz**。
  回傳格式與 GET /rag/rags、GET /exam/exams 完全一致；另可帶 weakness_report（AI 產生的 Markdown 弱點報告）。
  LLM API Key 由系統設定取得；若有設定則會依題目／參考答案／使用者答案彙整弱點，由 AI 產生報告。
"""

# 引入 json 用於解析 answer_feedback_metadata
import json
# 引入 Optional 型別
from typing import Optional

# 引入 FastAPI 的 APIRouter、HTTPException、PathParam
from fastapi import APIRouter, HTTPException, Path as PathParam
# 引入 OpenAI 客戶端
from openai import OpenAI
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 從 exam 模組引入共用查詢函數
from routers.exam import _answers_by_exam_quiz_ids, _exams_by_ids, _quizzes_by_person_id
# 引入 to_json_safe
from utils.json_utils import to_json_safe
# 引入系統 LLM API Key
from utils.llm_api_key_utils import get_llm_api_key

# 建立路由
router = APIRouter(prefix="/person-analysis", tags=["person analysis"])


class ListQuizzesByPersonResponse(BaseModel):
    """GET /person-analysis/quizzes/{person_id} 回應。格式與 GET /rag/rags、GET /exam/exams 一致；可選帶 weakness_report。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(default=None, description="依題目／參考答案／使用者答案彙整後由 AI 產生的 Markdown 弱點報告；系統未設定 LLM API Key 時為 None")


def _collect_weaknesses_from_quizzes(quizzes: list[dict]) -> list[str]:
    """從 person-analysis 回傳的 quizzes 中，收集所有 answer_feedback_metadata 的 weaknesses。"""
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


def _generate_weakness_report_md(quizzes: list[dict], api_key: str) -> Optional[str]:
    """依題目／參考答案／使用者答案彙整弱點，呼叫 LLM 產生 Markdown 報告。無弱點或 API 失敗時回傳 None。"""
    all_weaknesses = _collect_weaknesses_from_quizzes(quizzes)
    if not all_weaknesses:
        return None
    weakness_text = "\n".join(all_weaknesses[:60])
    prompt = f"""你是教學顧問。請根據以下來自測驗回饋的學習弱點，製作一份 Markdown 報告。
弱點列表：
{weakness_text}
                【重要限制】
                1. **請務必使用繁體中文 (Traditional Chinese) 撰寫所有評語、優點、弱點與行動建議。**
                輸出 JSON】{{ "簡介": [],  "學習弱點分析": [],  "建議": [],  "結論": [],  }}

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


@router.get("/quizzes/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id 取得該使用者在 Exam_Quiz 的資料，**僅回傳在 Exam_Answer 有對應答案的 quiz**。
    回傳格式與 GET /rag/rags、GET /exam/exams 完全一致；另帶 weakness_report（系統有設定 LLM API Key 時由 AI 產生）。
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
            raw_answers = (answers_by_quiz.get(qid_int, []) or []) if qid_int is not None else []
            quiz["answers"] = raw_answers[:1]
        # 只回傳在 Exam_Answer 有對應答案的 quiz
        quizzes_with_answers = [q for q in quizzes if (q.get("answers") or [])]
        exam_ids: list[int] = []
        for q in quizzes_with_answers:
            eid = q.get("exam_id")
            if eid is not None:
                try:
                    exam_ids.append(int(eid))
                except (TypeError, ValueError):
                    pass
        exam_ids = list(dict.fromkeys(exam_ids))
        exam_rows = _exams_by_ids(exam_ids)
        quizzes_by_exam: dict[int, list[dict]] = {eid: [] for eid in exam_ids}
        for q in quizzes_with_answers:
            eid = q.get("exam_id")
            if eid is not None:
                try:
                    quizzes_by_exam.setdefault(int(eid), []).append(q)
                except (TypeError, ValueError):
                    pass
        for row in exam_rows:
            eid = row.get("exam_id")
            eid_int = int(eid) if eid is not None else None
            row_quizzes = quizzes_by_exam.get(eid_int, []) if eid_int is not None else []
            row["quizzes"] = row_quizzes
            row["answers"] = []
            for q in row_quizzes:
                row["answers"].extend(q.get("answers") or [])
        data = to_json_safe(exam_rows)
        weakness_report: Optional[str] = None
        api_key = get_llm_api_key()
        if api_key:
            weakness_report = _generate_weakness_report_md(to_json_safe(quizzes_with_answers), api_key)
        return ListQuizzesByPersonResponse(exams=data, count=len(data), weakness_report=weakness_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
