"""
弱點報告 LLM 產生（個人分析、課程分析共用）。
對齊 `services/quiz_generation`、`services/grading`：system／user Markdown prompt、`.format(analysis_user_prompt_text=…, material_md=…)`。
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Optional

from openai import OpenAI

WEAKNESS_LLM_MODEL = "gpt-4o"


def _system_prompt_weakness_report(analysis_label: str) -> str:
    """analysis_label 例：個人分析、課程分析。"""
    section = f"## {analysis_label} user prompt"
    return textwrap.dedent(f"""
    # 角色

    你是教學顧問，請依使用者訊息中的測驗素材與 **`{section}`** 指令，以 Markdown 輸出教學分析與建議（勿預設套用固定報告名稱作為開頭標題，見下方「產出格式」）。

    ## 指令優先級（必須遵守）

    - 使用者訊息中 **`{section}`** 為管理員設定的**直接指令**（報告語氣、結構、弱點聚焦、篇幅等），優先級**高於**下方 **測驗素材** 與本 system 之泛化規則。
    - 該節有**實質文字**時（非僅空白或占位「（未提供）」），**必須完整遵守**，不得忽略、弱化或改寫其意圖。
    - 該節無實質文字時，始依 **測驗素材** 與本訊息其餘規範撰寫。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---` 等）。
    - 將 `---` 視為區段分隔。

    ## 產出格式

    - 請輸出**純 Markdown** 本文（**勿**使用 JSON）。
    - 除非 **{analysis_label} user prompt** 明確要求某標題或開頭字樣，請勿習慣性使用「學習弱點報告」等固定套用語作為粗體或一級標題；可直接進入內容，或使用與素材／指令相符的標題。
    - 僅輸出本文，不要前言後語或以程式碼區塊包整份內容。
    """).strip()


def _user_prompt_weakness_report(analysis_label: str) -> str:
    section = f"## {analysis_label} user prompt"
    return textwrap.dedent(f"""
    ## 必須遵守（最高優先）

    - 下節 **`{section}`** 之內文取自系統設定（與出題 **`## 出題 user prompt`**、批改之出題／作答 user prompt 區塊相同的「教師／管理員直接指令」角色）；與下方 **測驗素材** 牴觸時，**以該節為準**。
    - 該節有實質文字時**務必落實**；無實質文字（含僅「（未提供）」）時，請依 **測驗素材** 分析並輸出建議。
    - **測驗素材** 可能為**批改回饋列表**，也可能為尚無結構化評語時之**題幹、參考答案、學生作答與 quiz_rate** 摘要。

    {section}

    {{analysis_user_prompt_text}}

    ---

    ## 測驗素材

    {{material_md}}
    """).strip()


def quiz_has_answer(quiz: dict) -> bool:
    """有作答內容才納入弱點分析（避免空列干擾 LLM）。"""
    return bool((quiz.get("answer_content") or "").strip())


def analysis_user_prompt_display(raw: Optional[str]) -> str:
    """填入弱點報告 user 模板；空則與批改 `_grade_field_display` 一致為「（未提供）」。"""
    return (raw or "").strip() or "（未提供）"


def _synthetic_answer_from_quiz(quiz: dict) -> dict:
    return {
        "exam_quiz_id": quiz.get("exam_quiz_id"),
        "quiz_answer": quiz.get("answer_content"),
        "answer_critique": quiz.get("answer_critique"),
    }


def _metadata_for_weaknesses(ans: dict) -> Any:
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
    lines: list[str] = []
    w = data.get("weaknesses")
    if isinstance(w, list):
        lines.extend(s for s in w if isinstance(s, str) and s.strip())
    lines.extend(_strings_from_quiz_comments_field(data.get("quiz_comments")))
    return lines


def _collect_weaknesses_from_quizzes(quizzes: list[dict]) -> list[str]:
    all_weaknesses: list[str] = []
    for quiz in quizzes or []:
        ans = _synthetic_answer_from_quiz(quiz)
        meta = _metadata_for_weaknesses(ans)
        if not meta or not isinstance(meta, dict):
            continue
        all_weaknesses.extend(_feedback_lines_from_metadata_dict(meta))
    return all_weaknesses


def _clip(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _exam_quiz_rate_display(quiz: dict) -> str:
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


def generate_weakness_report_md(
    quizzes: list[dict],
    api_key: str,
    analysis_user_prompt_text: str,
    *,
    analysis_label: str,
) -> Optional[str]:
    """
    呼叫 LLM 產生弱點報告 Markdown。
    analysis_label：個人分析、課程分析（對應 user prompt 區段標題與 system 說明）。
    """
    if not (quizzes or []):
        return None

    feedback_lines = _collect_weaknesses_from_quizzes(quizzes)
    if feedback_lines:
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

    user_template = _user_prompt_weakness_report(analysis_label)
    user_content = user_template.format(
        analysis_user_prompt_text=analysis_user_prompt_display(analysis_user_prompt_text),
        material_md=material_md,
    )
    client = OpenAI(api_key=api_key)
    try:
        r = client.chat.completions.create(
            model=WEAKNESS_LLM_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt_weakness_report(analysis_label)},
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
