"""
個人分析 API 模組。
依 person_id 查詢 Exam_Quiz 資料。新 schema 答案欄位直接內嵌於 Exam_Quiz（answer_content, answer_critique），不再有獨立的 Exam_Answer 表。
- GET /person-analysis/quizzes/{person_id}：依 person_id 取得已作答的 Exam_Quiz（answer_content 非空），
  依 exam_tab_id 分群回傳；另帶 weakness_report（系統有 LLM API Key 時由 AI 產生）。

重要：弱點報告 prompt 與回應皆為 Markdown；與 json_object 出題／批改分流。

檔案結構：模型／檢索常數 → LLM Prompt → Pydantic → 弱點資料彙整私有函式 → 路由。
"""

import json
import textwrap
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Path as PathParam

from dependencies.person_id import PersonId
from openai import OpenAI
from pydantic import BaseModel, Field

from services.exam_queries import exams_by_tab_ids, quizzes_by_person_id
from utils.json_utils import to_json_safe
from utils.llm_api_key_utils import get_llm_api_key

router = APIRouter(prefix="/person-analysis", tags=["person analysis"])


# ---------------------------------------------------------------------------
# 模型與檢索常數
# ---------------------------------------------------------------------------

WEAKNESS_LLM_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# LLM Prompt 範本（弱點報告；集中維護）
# ---------------------------------------------------------------------------
# {material}：由 _generate_weakness_report_md 動態組成（批改評語列表或試卷摘要），勿在常數內寫死內容。

PROMPT_WEAKNESS_REPORT = textwrap.dedent("""
    # 任務

    你是教學顧問。請根據以下測驗資料，撰寫一份**純 Markdown** 學習弱點報告（**勿**使用 JSON）。

    ---

    {material}

    ---

    ## 撰寫要求

    1. 結構請包含：**簡介**、**學習弱點分析**、**具體建議**、**結論**（使用 Markdown 標題與條列即可）。
    2. 僅輸出報告本文，不要前言後語或以程式碼區塊包整份報告。
    """).strip()


# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------

class ListQuizzesByPersonResponse(BaseModel):
    """GET /person-analysis/quizzes/{person_id} 回應。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="依題目與作答由 AI 產生的 Markdown 弱點報告；系統有設定 LLM API Key 時必為非空字串；未設定時為 null",
    )


# ---------------------------------------------------------------------------
# 弱點報告：資料彙整（私有）
# ---------------------------------------------------------------------------

def _quiz_has_answer(quiz: dict) -> bool:
    """有作答內容才納入弱點分析（避免空列干擾 LLM）。"""
    return bool((quiz.get("answer_content") or "").strip())


def _synthetic_answer_from_quiz(quiz: dict) -> dict:
    """從 Exam_Quiz 的內嵌欄位構造 answer 摘要（供弱點分析使用）。"""
    return {
        "exam_quiz_id": quiz.get("exam_quiz_id"),
        "quiz_answer": quiz.get("answer_content"),
        "answer_critique": quiz.get("answer_critique"),
    }


def _metadata_for_weaknesses(ans: dict) -> Any:
    """解析 answer_critique：JSON 物件、JSON 字串，或純文字評語（新寫入格式）。"""
    meta = ans.get("answer_critique")
    if not meta:
        return None
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        s = meta.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {"quiz_comments": [s]}
        else:
            return parsed if isinstance(parsed, dict) else {"quiz_comments": [s]}
    return None


def _strings_from_quiz_comments_field(raw: Any) -> list[str]:
    """將 critique 內 quiz_comments（多型別元素）展平為字串列表。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        elif isinstance(x, dict):
            c = x.get("quiz_comment") or x.get("comment") or x.get("criteria")
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
    """合併 weaknesses 陣列與 quiz_comments 內可讀字串，供弱點報告素材。"""
    lines: list[str] = []
    w = data.get("weaknesses")
    if isinstance(w, list):
        lines.extend(s for s in w if isinstance(s, str) and s.strip())
    lines.extend(_strings_from_quiz_comments_field(data.get("quiz_comments")))
    return lines


def _collect_weaknesses_from_quizzes(quizzes: list[dict]) -> list[str]:
    """從 Exam_Quiz 的 answer_critique 收集弱點評語。"""
    all_weaknesses: list[str] = []
    for quiz in quizzes or []:
        ans = _synthetic_answer_from_quiz(quiz)
        meta = _metadata_for_weaknesses(ans)
        if not meta or not isinstance(meta, dict):
            continue
        all_weaknesses.extend(_feedback_lines_from_metadata_dict(meta))
    return all_weaknesses


def _clip(text: str, max_len: int) -> str:
    """截斷過長欄位，避免 PROMPT token 膨脹（長度與 _build_quiz_context_block 呼叫處一致）。"""
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _exam_quiz_rate_display(quiz: dict) -> str:
    """Exam_Quiz 已無數值得分欄位；以 `quiz_rate`（常見為 -1／0／1）供弱點分析脈絡。"""
    raw = quiz.get("quiz_rate")
    if raw is None:
        return "（無 quiz_rate）"
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return str(raw)
    by_val = {
        -1: "-1（負向標記）",
        0: "0（預設／中性）",
        1: "1（正向標記）",
    }
    return by_val.get(v, f"{v}")


def _build_quiz_context_block(quizzes: list[dict]) -> str:
    """無結構化評語時，以題幹／參考答案／作答／quiz_rate 組 Markdown 區塊供 LLM 分析。"""
    parts: list[str] = []
    for i, quiz in enumerate(quizzes or [], 1):
        q = _clip(str(quiz.get("quiz_content") or ""), 800)
        ref = _clip(str(quiz.get("quiz_answer_reference") or ""), 600)
        ua = _clip(str(quiz.get("answer_content") or ""), 800)
        rate = _exam_quiz_rate_display(quiz)
        parts.append(
            textwrap.dedent(f"""
                ### 第 {i} 題

                - **題目**：{q or "（無）"}
                - **參考答案**：{ref or "（無）"}
                - **學生作答**：{ua or "（無）"}
                - **評級（quiz_rate）**：{rate}
                """).strip()
        )
    return "\n\n".join(parts) if parts else ""


def _generate_weakness_report_md(quizzes: list[dict], api_key: str) -> str:
    """呼叫 LLM 產生 Markdown 弱點報告；失敗或無 key 時回傳內建 fallback 字串。"""
    fallback_llm = (
        "## 學習弱點報告\n\n"
        "AI 服務暫時無法產生報告，請稍後再試。您仍可依各題的 quiz_rate 與評語於下方試卷資料檢視。"
    )
    if not (quizzes or []):
        return "## 學習弱點報告\n\n目前沒有已作答的題目，無法彙整弱點。"

    feedback_lines = _collect_weaknesses_from_quizzes(quizzes)
    if feedback_lines:
        # 優先使用 critique 內結構化弱點／評語（最多 80 條控制長度）。
        material = "## 批改回饋與評語\n\n" + "\n".join(f"- {line}" for line in feedback_lines[:80])
    else:
        # 無評語時退回試卷原文摘要，仍要求模型產出弱點報告。
        ctx = _build_quiz_context_block(quizzes)
        material = textwrap.dedent(f"""
            ## 試卷與作答摘要

            以下為各題題幹、參考答案、學生作答與評級 quiz_rate（尚無結構化評語時請依此分析學習弱點）：

            {ctx or "（無可分析內容）"}
            """).strip()

    prompt = PROMPT_WEAKNESS_REPORT.format(material=material)
    client = OpenAI(api_key=api_key)
    try:
        # 弱點報告為自由格式 Markdown，不使用 response_format=json_object。
        r = client.chat.completions.create(
            model=WEAKNESS_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        out = (r.choices[0].message.content or "").strip()
        if out:
            return out
    except Exception:
        pass
    return fallback_llm


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.get("/quizzes/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    caller_person_id: PersonId,
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id 取得已作答的 Exam_Quiz（answer_content 非空），依 exam_tab_id 分群後對應 Exam。
    另帶 weakness_report（系統設定 LLM API Key 時由 AI 產生）。
    query 的 person_id 須與路徑 {person_id} 一致。
    """
    try:
        path_pid = (person_id or "").strip()
        if path_pid != caller_person_id:
            raise HTTPException(status_code=403, detail="路徑 person_id 與 query 不一致")

        quizzes = quizzes_by_person_id(path_pid)
        quizzes_with_answers = [q for q in quizzes if _quiz_has_answer(q)]

        tab_ids: list[str] = list(dict.fromkeys(
            str(q.get("exam_tab_id")) for q in quizzes_with_answers if q.get("exam_tab_id") is not None
        ))
        exam_rows = exams_by_tab_ids(tab_ids)
        quizzes_by_tab: dict[str, list[dict]] = {tid: [] for tid in tab_ids}
        for q in quizzes_with_answers:
            tid = q.get("exam_tab_id")
            if tid is not None:
                quizzes_by_tab.setdefault(str(tid), []).append(q)

        for row in exam_rows:
            tid = str(row.get("exam_tab_id") or "")
            row_quizzes = quizzes_by_tab.get(tid, [])
            row["quizzes"] = row_quizzes
            row["answers"] = [_synthetic_answer_from_quiz(q) for q in row_quizzes]

        data = to_json_safe(exam_rows)
        weakness_report: Optional[str] = None
        api_key = get_llm_api_key()
        if api_key:
            weakness_report = _generate_weakness_report_md(to_json_safe(quizzes_with_answers), api_key)
        return ListQuizzesByPersonResponse(exams=data, count=len(data), weakness_report=weakness_report)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
