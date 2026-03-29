"""
個人分析 API 模組。
依 person_id 查詢 Exam_Quiz / Exam_Answer 等分析用資料。
- GET /person-analysis/quizzes/{person_id}：依 person_id 取得該使用者在 Exam_Quiz 的資料，**僅回傳在 Exam_Answer 有對應答案的 quiz**。
  回傳格式與 GET /rag/tabs、GET /exam/tabs 完全一致；另帶 weakness_report（系統有設定 LLM API Key 時必為非空字串，AI 產生 Markdown 或備援說明）。
  LLM API Key 由 System_Setting 取得；會優先使用批改 metadata 的 quiz_comments／weaknesses，否則以題幹與作答呼叫 AI 彙整。
"""

# 引入 json 用於解析 quiz_grade_metadata／answer_feedback_metadata
import json
# 引入 Optional 型別
from typing import Any, Optional

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
from utils.http_retry import call_with_500_retry

# 建立路由
router = APIRouter(prefix="/person-analysis", tags=["person analysis"])


class ListQuizzesByPersonResponse(BaseModel):
    """GET /person-analysis/quizzes/{person_id} 回應。格式與 GET /rag/tabs、GET /exam/tabs 一致；可選帶 weakness_report。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="依題目／參考答案／使用者答案與評分回饋彙整後由 AI 產生的 Markdown 弱點報告；"
        "系統已設定 LLM API Key 時必為非空字串（AI 失敗時為備援說明）；未設定 Key 時為 null",
    )


def _metadata_for_weaknesses(ans: dict) -> Any:
    """quiz_grade_metadata；舊列可能仍為 answer_feedback_metadata。"""
    meta = ans.get("quiz_grade_metadata") or ans.get("answer_feedback_metadata")
    return meta


def _strings_from_quiz_comments_field(raw: Any) -> list[str]:
    """與評分寫入的 quiz_comments 對齊：字串列表或物件列表（quiz_comment / comment / criteria）。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        elif isinstance(x, dict):
            c = x.get("quiz_comment")
            if c is None:
                c = x.get("comment")
            if c is None:
                c = x.get("criteria")
            if c is not None:
                s = str(c).strip()
                if s:
                    out.append(s)
        elif x is not None:
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _feedback_lines_from_metadata_dict(data: dict) -> list[str]:
    """從單筆 metadata 取出弱點相關文字（weaknesses 舊欄位 + 評分實際寫入的 quiz_comments）。"""
    lines: list[str] = []
    w = data.get("weaknesses")
    if isinstance(w, list):
        lines.extend(s for s in w if isinstance(s, str) and s.strip())
    lines.extend(_strings_from_quiz_comments_field(data.get("quiz_comments")))
    return lines


def _collect_weaknesses_from_quizzes(quizzes: list[dict]) -> list[str]:
    """從 quizzes 的 answers metadata 收集弱點／評語文字（含 quiz_grade_metadata.quiz_comments）。"""
    all_weaknesses: list[str] = []
    for quiz in quizzes or []:
        answers = quiz.get("answers") or []
        for ans in answers:
            meta = _metadata_for_weaknesses(ans)
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
            if not isinstance(data, dict):
                continue
            all_weaknesses.extend(_feedback_lines_from_metadata_dict(data))
    return all_weaknesses


def _clip(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _build_quiz_context_block(quizzes: list[dict]) -> str:
    """無評語時改以題目、參考答案、作答、分數供 LLM 彙整弱點。"""
    parts: list[str] = []
    for i, quiz in enumerate(quizzes or [], 1):
        q = _clip(str(quiz.get("quiz_content") or ""), 800)
        ref = _clip(
            str(
                quiz.get("quiz_reference_answer")
                or quiz.get("quiz_answer_reference")
                or ""
            ),
            600,
        )
        for ans in (quiz.get("answers") or [])[:1]:
            ua = _clip(str(ans.get("quiz_answer") or ans.get("answer") or ""), 800)
            grade = ans.get("quiz_grade")
            parts.append(
                f"【第 {i} 題】\n題目：{q or '（無）'}\n參考答案：{ref or '（無）'}\n學生作答：{ua or '（無）'}\n得分：{grade!s}"
            )
    return "\n\n".join(parts) if parts else ""


def _generate_weakness_report_md(quizzes: list[dict], api_key: str) -> str:
    """已設定 API Key 時必回傳非空 Markdown 字串（LLM 失敗則為備援說明）。"""
    fallback_llm = (
        "## 學習弱點報告\n\n"
        "AI 服務暫時無法產生報告，請稍後再試。您仍可依各題的評分與評語於下方試卷資料檢視。"
    )
    if not (quizzes or []):
        return "## 學習弱點報告\n\n目前沒有已作答的題目，無法彙整弱點。"

    feedback_lines = _collect_weaknesses_from_quizzes(quizzes)
    if feedback_lines:
        material = "來自測驗批改的回饋與評語：\n" + "\n".join(feedback_lines[:80])
    else:
        ctx = _build_quiz_context_block(quizzes)
        material = (
            "以下為各題題幹、參考答案、學生作答與得分（尚無結構化評語時請依此分析學習弱點）：\n\n"
            + (ctx or "（無可分析內容）")
        )

    prompt = f"""你是教學顧問。請根據以下測驗資料，撰寫一份**純 Markdown** 學習弱點報告（勿使用 JSON）。

{material}

【重要】
1. 全文請使用**繁體中文**。
2. 結構請包含：簡介、學習弱點分析、具體建議、結論（使用 Markdown 標題與條列即可）。
3. 僅輸出報告本文，不要前言後語或程式碼區塊包整份報告。
"""
    client = OpenAI(api_key=api_key)
    try:
        r = call_with_500_retry(
            lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        )
        out = (r.choices[0].message.content or "").strip()
        if out:
            return out
    except Exception:
        pass
    return fallback_llm


@router.get("/quizzes/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id 取得該使用者在 Exam_Quiz 的資料，**僅回傳在 Exam_Answer 有對應答案的 quiz**。
    回傳格式與 GET /rag/tabs、GET /exam/tabs 完全一致；另帶 weakness_report（系統有設定 LLM API Key 時由 AI 產生）。
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
