"""
個人分析 API 模組。
依 person_id 查詢 Exam_Quiz 資料。新 schema 答案欄位直接內嵌於 Exam_Quiz（answer_content, answer_critique），不再有獨立的 Exam_Answer 表。
- GET /person-analysis/quizzes/{person_id}：依 person_id 取得已作答的 Exam_Quiz（answer_content 非空），
  依 exam_tab_id 分群回傳 Exam；每筆 Exam 的題目結構與 GET /exam/tabs 相同（units[]，依 unit_name 分群之 Exam_Quiz，含 enrich／rag 鍵）。
  另帶 weakness_report：**預設不呼叫 LLM**，`weakness_report` 為 null；僅當 query **`generate_weakness_report=true`** 時才產生弱點報告（有 LLM API Key 且成功呼叫時為模型回覆原文，否則 null）。

重要：弱點報告與出題／批改相同，系統與使用者訊息皆為 **Markdown**；本路由**不**使用 `response_format=json_object`。
**個人分析 user prompt** 取自 `System_Setting.key=person_analysis_user_prompt_text`（與 GET/PUT `/system-settings/person_analysis_user_prompt_text` 同源），嵌入 user 訊息之 **`## 個人分析 user prompt`**，優先級對齊出題之 **`## 出題 user prompt`**、批改之出題／作答 user prompt 區塊。

檔案結構（由上而下）：
1. 模型常數（`WEAKNESS_LLM_MODEL`，與出題 `QUIZ_LLM_MODEL`、批改 `GRADE_LLM_MODEL` 對齊為同一型號字串）
2. **LLM 弱點報告 Prompt**（對齊 `utils/quiz_generation`、`services/grading`：`SYSTEM_PROMPT_WEAKNESS_REPORT`、`USER_PROMPT_WEAKNESS_REPORT`；user 以 `.format(person_analysis_user_prompt_text=…, material_md=…)` 填入）
3. Pydantic → 弱點資料彙整私有函式 → 路由。
"""

import json
import textwrap
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Path as PathParam, Query

from dependencies.person_id import PersonId
from openai import OpenAI
from pydantic import BaseModel, Field

from services.exam_queries import (
    exams_by_tab_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    group_exam_quizzes_into_units,
    quizzes_by_person_id,
)
from routers.system_settings import SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY
from utils.json_utils import to_json_safe
from utils.llm_api_key_utils import get_llm_api_key
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/person-analysis", tags=["person analysis"])


# -----------------------------------------------------------------------------
# 模型常數
# -----------------------------------------------------------------------------
# 與 `utils.quiz_generation.QUIZ_LLM_MODEL`、`services.grading.GRADE_LLM_MODEL` 一致，便於維運對照。

WEAKNESS_LLM_MODEL = "gpt-4o"


# -----------------------------------------------------------------------------
# LLM 弱點報告 Prompt（對齊 `quiz_generation`／`grading`：SYSTEM_PROMPT_* → USER_PROMPT_*、`---`、`## …` 分段；user 僅 `.format(...)`）
# -----------------------------------------------------------------------------
# Chat messages：
#   role=system … SYSTEM_PROMPT_WEAKNESS_REPORT
#   role=user … USER_PROMPT_WEAKNESS_REPORT；其中 `{person_analysis_user_prompt_text}` 取自 System_Setting，
#   `{material_md}` 由 _generate_weakness_report_md 組入（批改回饋條列或試卷／作答摘要擇一）。
#

SYSTEM_PROMPT_WEAKNESS_REPORT = textwrap.dedent("""
    # 角色

    你是教學顧問，請依使用者訊息中的測驗素材與 **`## 個人分析 user prompt`** 指令，以 Markdown 輸出教學分析與建議（勿預設套用固定報告名稱作為開頭標題，見下方「產出格式」）。

    ## 指令優先級（必須遵守）

    - 使用者訊息中 **`## 個人分析 user prompt`** 為管理員設定的**直接指令**（報告語氣、結構、弱點聚焦、篇幅等），優先級**高於**下方 **測驗素材** 與本 system 之泛化規則。
    - 該節有**實質文字**時（非僅空白或占位「（未提供）」），**必須完整遵守**，不得忽略、弱化或改寫其意圖。
    - 該節無實質文字時，始依 **測驗素材** 與本訊息其餘規範撰寫。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---` 等）。
    - 將 `---` 視為區段分隔。

    ## 產出格式

    - 請輸出**純 Markdown** 本文（**勿**使用 JSON）。
    - 除非 **個人分析 user prompt** 明確要求某標題或開頭字樣，請勿習慣性使用「學習弱點報告」等固定套用語作為粗體或一級標題；可直接進入內容，或使用與素材／指令相符的標題。
    - 僅輸出本文，不要前言後語或以程式碼區塊包整份內容。
    """).strip()

USER_PROMPT_WEAKNESS_REPORT = textwrap.dedent("""
    ## 必須遵守（最高優先）

    - 下節 **`## 個人分析 user prompt`** 之內文取自系統設定（與出題 **`## 出題 user prompt`**、批改之出題／作答 user prompt 區塊相同的「教師／管理員直接指令」角色）；與下方 **測驗素材** 牴觸時，**以該節為準**。
    - 該節有實質文字時**務必落實**；無實質文字（含僅「（未提供）」）時，請依 **測驗素材** 分析並輸出建議。
    - **測驗素材** 可能為**批改回饋列表**，也可能為尚無結構化評語時之**題幹、參考答案、學生作答與 quiz_rate** 摘要。

    ## 個人分析 user prompt

    {person_analysis_user_prompt_text}

    ---

    ## 測驗素材

    {material_md}
    """).strip()


# -----------------------------------------------------------------------------
# Pydantic 模型
# -----------------------------------------------------------------------------

class ListQuizzesByPersonResponse(BaseModel):
    """GET /person-analysis/quizzes/{person_id} 回應。exams[] 每筆與 GET /exam/tabs 相同含 units[]；weakness_report 僅於 generate_weakness_report=true 時才可能非 null。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：僅於 query generate_weakness_report=true 時才可能產生；為 LLM `message.content` 原文；未請求產生、未設定 API Key、呼叫失敗或無內容時為 null",
    )


# -----------------------------------------------------------------------------
# 弱點報告：資料彙整（私有）
# -----------------------------------------------------------------------------

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


def _person_analysis_user_prompt_display(raw: Optional[str]) -> str:
    """填入弱點報告 user 模板之「個人分析 user prompt」欄位；空則與批改 `_grade_field_display` 一致為「（未提供）」。"""
    return (raw or "").strip() or "（未提供）"


def _fetch_person_analysis_user_prompt_text_from_setting() -> str:
    """讀取 System_Setting key=person_analysis_user_prompt_text；失敗或無列時回傳空字串。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return ""
        return (resp.data[0].get("value") or "").strip()
    except Exception:
        return ""


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


def _generate_weakness_report_md(quizzes: list[dict], api_key: str) -> Optional[str]:
    """呼叫 LLM；成功時只回傳 `choices[0].message.content` 原文（不 strip、不補前後文）。無題可分析、失敗或無內容時回傳 None。"""
    if not (quizzes or []):
        return None

    feedback_lines = _collect_weaknesses_from_quizzes(quizzes)
    if feedback_lines:
        # 優先使用 critique 內結構化弱點／評語（最多 80 條）；併入 USER_PROMPT_WEAKNESS_REPORT 之 {material_md}。
        material_md = "## 批改回饋與評語\n\n" + "\n".join(
            f"- {line}" for line in feedback_lines[:80]
        )
    else:
        ctx = _build_quiz_context_block(quizzes)
        material_md = textwrap.dedent(f"""
            ## 試卷與作答摘要

            以下為各題題幹、參考答案、學生作答與評級 quiz_rate（尚無結構化評語時請依此分析作答表現）：

            {ctx or "（無可分析內容）"}
            """).strip()

    setting_prompt = _fetch_person_analysis_user_prompt_text_from_setting()
    user_content = USER_PROMPT_WEAKNESS_REPORT.format(
        person_analysis_user_prompt_text=_person_analysis_user_prompt_display(setting_prompt),
        material_md=material_md,
    )
    client = OpenAI(api_key=api_key)
    try:
        # 弱點報告為自由格式 Markdown，不使用 response_format=json_object。
        r = client.chat.completions.create(
            model=WEAKNESS_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_WEAKNESS_REPORT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        msg = r.choices[0].message
        if msg is None:
            return None
        return msg.content
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 路由
# -----------------------------------------------------------------------------

@router.get("/quizzes/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    caller_person_id: PersonId,
    person_id: str = PathParam(..., description="要查詢的 person_id"),
    generate_weakness_report: bool = Query(
        False,
        description="為 true 時才呼叫 LLM 產生 weakness_report；預設 false（僅載入試卷資料，與「儲存 prompt」分離）",
    ),
):
    """
    依 person_id 取得已作答的 Exam_Quiz（answer_content 非空），依 exam_tab_id 分群後對應 Exam；
    每筆 Exam 的 units／quizzes 形狀與 GET /exam/tabs 一致（題目為完整 Exam_Quiz 列，含作答欄位）。
    weakness_report：僅當 `generate_weakness_report=true` 時才可能產生；成功時為 LLM `message.content` 原文；
    否則為 null。無 API Key、無可分析題目、呼叫失敗或模型回 null 時亦為 null。
    弱點報告 user 訊息會併入 System_Setting `person_analysis_user_prompt_text`
    （與 `/system-settings/person_analysis_user_prompt_text` 同源）。
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

        flat_for_enrich = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_for_enrich)
        ensure_exam_quiz_rag_id_keys(flat_for_enrich)

        for row in exam_rows:
            tid = str(row.get("exam_tab_id") or "")
            row["units"] = group_exam_quizzes_into_units(quizzes_by_tab.get(tid, []))

        data = to_json_safe(exam_rows)
        weakness_report: Optional[str] = None
        if generate_weakness_report:
            api_key = get_llm_api_key()
            if api_key:
                weakness_report = _generate_weakness_report_md(to_json_safe(quizzes_with_answers), api_key)
        return ListQuizzesByPersonResponse(exams=data, count=len(data), weakness_report=weakness_report)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
