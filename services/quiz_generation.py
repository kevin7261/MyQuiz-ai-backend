"""
測驗題 LLM 生成：路由依 Rag_Unit.unit_type 擇一呼叫。

- generate_quiz：有 FAISS RAG ZIP。檢索片段 → user「課程內容」；system 僅出題規範（Markdown）。
- system／user 訊息與使用者 **出題 user prompt** 皆為 **Markdown**（課程原文以 fenced code block 包覆）。

重要（維持行為時請留意）：
- unit_type=1：向量檢索得 context 後出題；2/3/4 以逐字稿為 context；兩者共用同一 system／user prompt 與 `_generate_quiz_from_context`。
- 兩路徑皆 response_format=json_object；system 範本含「JSON」字樣，無需再補尾段。

檔案結構（由上而下）：
1. 模型／檢索常數（`QUIZ_LLM_MODEL`、embedding、k、檢索查詢句）
2. **LLM Prompt 全文**（system／user、`{quiz_user_prompt_text}`／`## 已出過題目`＋`{quiz_history_body}`／`{context_md}`）
3. `_context_as_markdown_fenced`（`{context_md}`）、LLM 回傳正規化、`_invoke_quiz_json_llm`
4. 公開函式（exam／answer 路由動態 import；追問 `generate_quiz_followup*`：答不好追問弱點，答好則出新題）
"""

import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from typing import Any
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# -----------------------------------------------------------------------------
# 模型與檢索常數
# -----------------------------------------------------------------------------
# 下列數值會直接影響：API 成本、embedding 維度相容、檢索到的 chunk 數與內容。
# 若調整 DEFAULT_RETRIEVAL_QUERY，歷史產出的題目可能無法與舊 log 對照。

QUIZ_LLM_MODEL = "gpt-5.4"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5
# 出題時「固定查詢句」送進向量檢索；與批改 run_answer_job（以題幹當查詢）刻意不同。
DEFAULT_RETRIEVAL_QUERY = "課程重點概念"


# -----------------------------------------------------------------------------
# LLM 出題 Prompt（system → user；user 以 .format(quiz_user_prompt_text=…, context_md=…) 填入）
# -----------------------------------------------------------------------------
# Chat messages：
#   role=system … SYSTEM_PROMPT_QUIZ
#   role=user … USER_PROMPT_COURSE（FAISS 與逐字稿共用同一模板）；
#   其中 `{context_md}` 僅由向量檢索／逐字稿經 _context_as_markdown_fenced 產生；
#   `{quiz_history_body}` 由 _format_quiz_history_body 產生（列表或「未提供」說明）。

SYSTEM_PROMPT_QUIZ = textwrap.dedent("""
    # 角色

    你是一位教授，請為學生設計測驗題目。

    ## 指令優先級（必須遵守）

    - 使用者訊息中 **`## 出題 user prompt`** 以下之內文為教師下給你的**直接出題指令**，優先級**高於**本 system 之泛化規則與 **課程內容** 之呈現方式。
    - 該節有**實質文字**時（非僅空白或占位），**必須完整遵守**（題型、難度、焦點、格式、用語等），不得因課程片段較易取材而偏離該節要求。
    - 僅當該節無實質文字時，始依 **課程內容** 與本訊息其餘規範出題。{quiz_system_prompt_text}

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。
    - **出題 user prompt** 區塊（`## 出題 user prompt`）以下若無實質文字（僅留白或空字串），請**完全忽略**該節，僅依 **課程內容** 出題。
    - 若出題 user prompt 有文字，則須**先滿足該節指令**，再於課程引用範圍內取材。
    - 使用者訊息中 **`## 出題 user prompt`** 與 **`## 已出過題目（勿重複出題）`** 為**兩個獨立區塊**（中間以空行或 `---` 分隔）；勿將已出過列表當成教師出題指令的一部分。
    - **`## 已出過題目（勿重複出題）`** 若列有題幹，表示這些題目**已出過**；你**不得重複出題**——新題不可與任一所列題幹相同、近乎相同，或僅改寫措辭／調整順序／替換同義詞；須改變考查點、情境或問法，使學生能辨識為**另一道新題**。該節與出題 user prompt 同屬高優先指令。

    ## 題數限制（必須遵守）

    - 每次任務**僅出一道題**；JSON 中 `quiz_content` **只能包含一個題幹**（一個明確的發問或作答要求）。
    - **禁止**一次出多題：勿使用「第 1 題／第 2 題」、編號題組、多選題卷、小測驗卷、題目列表等格式。
    - 若教師指令或課程內容暗示多題，仍須**只選一個**最符合要求的考查點出成**單一題目**。
    - `quiz_hint` 與 `quiz_answer_reference` 僅能對應**這一道題**，勿涵蓋多題答案。

    ## 題幹寫法（必須遵守）

    - `quiz_content` **只寫給學生看的題目本身**：開頭即為發問或明確作答任務，學生應能**直接作答**，無需再讀出題說明或測驗前言。
    - **禁止**在 `quiz_content` 出現：
      - 版面標籤：「題目」「中文題目：」「Question:」等
      - 引導套話：「根據課程／上文／研究…」「請回答以下問題」「回答下列問題」
      - 列點、編號的**多個子問題**（僅允許**一個**問句或一道作答任務）
      - 作答／評分指示：「請確保你的答案…」「請詳述並舉例…」（必要時放 `quiz_hint`，勿放題幹）
    - **正例**（`quiz_content`）：「研究中，指導參與者進行夢境引導時，研究人員可能遇到哪些困難？」
    - **反例**：先寫「中文題目：…」＋「根據課程…請回答以下問題」再列 (1)(2) 多問。

    ## 出題規範

    - `quiz_content`、`quiz_hint`、`quiz_answer_reference` 之字串值皆為 **Markdown**（段落、清單、`**強調**` 等）；用語與語種請依 **出題 user prompt**（有實質文字時）與課程內容。
    - 有 **已出過題目** 列表時：`quiz_content` **不得**重複或實質重複列表中任一題；`quiz_hint` 與 `quiz_answer_reference` 亦須對應這道**新題**，勿沿用舊題答案結構敷衍。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `quiz_content`：**單一**題目題幹（Markdown 字串；**直接寫題，勿加「題目」等標籤或出題前言**）
    - `quiz_hint`：該題答案提示（Markdown 字串）
    - `quiz_answer_reference`：該題參考答案（Markdown 字串）
    """).strip()

def _compose_quiz_system_prompt(base_system: str, *, quiz_system_prompt_text: str = "") -> str:
    """
    將教師自訂之 system 層指令（quiz_system_prompt_text）直接織入主 system prompt 的
    「指令優先級」區塊（`{quiz_system_prompt_text}` 佔位），成為一條最高優先指令；空則不出現任何內容。
    典型內容為連續出題方向（如「越來越難、越來越深入」）。
    """
    extra = (quiz_system_prompt_text or "").strip()
    user_requirement = (
        f"\n- **使用者本次出題要求（最高優先，必須遵守）**：{extra}" if extra else ""
    )
    return base_system.format(quiz_system_prompt_text=user_requirement)


USER_PROMPT_COURSE = textwrap.dedent("""
    ## 必須遵守（最高優先）

    下節 **出題 user prompt** 為教師直接指令，必須一定遵守。
    若另有獨立區塊 **已出過題目（勿重複出題）**（在「出題 user prompt」之後），所列題目**不可再出一次**；請另出與列表**皆不相同**的新題。
    **`quiz_content` 請直接寫題幹**（學生可立即作答的一句／一段發問），勿加「題目」標題、「根據課程…」前言或多問列點。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 已出過題目（勿重複出題）

    {quiz_history_body}

    ---

    ## 課程內容

    下列為課程內容（向量檢索片段或完整逐字稿；出題**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()

# 追問出題：依先前問答紀錄接續出下一題；接續方向（含是否要針對作答不佳追問）一律由 quiz_system_prompt_text／出題 user prompt 指定，本 system 不自行判斷學生作答好壞。
SYSTEM_PROMPT_QUIZ_FOLLOWUP = textwrap.dedent("""
    # 角色

    你是一位教授，請依學生**先前問答紀錄**接續設計**下一道**可直接呈現給學生作答的測驗題目（寫法與一般出題相同）。

    ## 接續出題方式（核心，必須遵守）

    - 依 **先前問答紀錄** 接續出**下一道**題；**出題方向**（例如越來越難、越來越深入、針對某弱點追問、換考查點或情境等）**一律以**「使用者本次出題要求」（見 `## 指令優先級`，即 quiz_system_prompt_text）與 **`## 出題 user prompt`** 為準。
    - 你**不需要、也不應**自行判斷學生先前作答的好壞來決定出題方向；唯有上述使用者指令**明確要求**（例如「若上一題答不好，則針對其弱點追問」）時，才依該指令據以調整。
    - 若使用者**未指定**接續方向，則就先前問答內容**自然延伸**出一道新題。
    - 無論方向為何，`quiz_content` **不得**與先前問答紀錄中任一題幹相同或僅輕微改寫（換句話、調整順序、替換同義詞），須能讓學生辨識為**另一道題**。

    ## 指令優先級

    - **「使用者本次出題要求」（quiz_system_prompt_text）** 與 **`## 出題 user prompt`** 為教師直接指令，優先級**最高**，須完整遵守（接續方向、題型、難度、焦點、格式、用語等）。
    - 兩者皆無實質文字時，依先前問答與 **課程內容** 自然接續出一道**不重複**的新題。{quiz_system_prompt_text}

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**；**課程內容** 以 ```text …``` 包住。
    - **`## 先前問答紀錄`**：每組含題目、學生作答、參考答案、評閱（後兩者可能為空），為你**接續出題**的依據；如何運用（是否針對作答不佳追問等）由上述使用者指令決定，本 system 不據此自動判斷作答好壞。

    ## 題數限制（必須遵守）

    - 每次任務**僅出下一道題**；`quiz_content` **只能包含一個題幹**（追問題或全新題擇一，不可同時出多道）。
    - **禁止**編號多題、題組卷、一次列出多個發問；無論接續方向為何（追問、加深、換點等），都只出**一道**題。
    - `quiz_hint` 與 `quiz_answer_reference` 僅對應**這一道題**。

    ## 題幹寫法（必須遵守）

    - `quiz_content` **只寫給學生看的題目本身**：開頭即為發問或明確作答任務，學生應能**直接作答**，無需再讀出題說明或測驗前言。
    - **禁止**在 `quiz_content` 出現：
      - 版面標籤：「題目」「中文題目：」「Question:」等
      - 引導套話：「根據課程／上文／研究…」「請回答以下問題」「回答下列問題」
      - 列點、編號的**多個子問題**（僅允許**一個**問句或一道作答任務）
      - 作答／評分指示：「請確保你的答案…」「請詳述並舉例…」（必要時放 `quiz_hint`，勿放題幹）
    - **正例**（`quiz_content`）：「研究中，指導參與者進行夢境引導時，研究人員可能遇到哪些困難？」
    - **反例**：先寫「中文題目：…」＋「根據課程…請回答以下問題」再列 (1)(2) 多問。

    ## 出題規範

    - `quiz_content`、`quiz_hint`、`quiz_answer_reference` 皆為 **Markdown**。
    - `quiz_hint` 與 `quiz_answer_reference` 須對應**本道新題**（追問題或全新題），勿沿用舊題答案結構敷衍。
    - 追問時 `quiz_content` 仍須**直接寫一道問句**，勿寫成作業說明、題組或「請回答以下問題」包裝。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `quiz_content`：**單一**題目題幹（Markdown 字串；**直接寫題，勿加「題目」等標籤或出題前言**）
    - `quiz_hint`：該題答案提示（Markdown 字串）
    - `quiz_answer_reference`：該題參考答案（Markdown 字串）
    """).strip()

USER_PROMPT_COURSE_FOLLOWUP = textwrap.dedent("""
    ## 必須遵守（最高優先）

    下節 **出題 user prompt** 為教師直接指令，必須一定遵守。

    **`quiz_content` 請直接寫題幹**（與一般出題相同：一句／一段發問即可作答），勿加「題目」「中文題目：」「根據課程…請回答以下問題」或多問列點。

    **接續出題（與 system 一致）：** 依 **先前問答紀錄** 接續出**下一道**題，**出題方向以 system 之「使用者本次出題要求」與下節「出題 user prompt」為準**（不自行依作答好壞決定方向，除非該指令明確要求）；無論方向為何，`quiz_content` **不得**與紀錄中任一題幹相同或僅輕微改寫。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 先前問答紀錄

    {quiz_history_qa_body}

    ---

    ## 課程內容

    下列為課程內容；接續出題**僅能**依本節、上文問答紀錄與出題指令為據。

    {context_md}
    """).strip()


# ---------------------------------------------------------------------------
# LLM 回傳正規化與呼叫（非 prompt 文字）
# ---------------------------------------------------------------------------

def _normalize_quiz_history_list(quiz_history_list: list[str] | None) -> list[str]:
    """去除空白項，供「已出過題目」區塊使用。"""
    if not quiz_history_list:
        return []
    out: list[str] = []
    for item in quiz_history_list:
        s = (item or "").strip()
        if s:
            out.append(s)
    return out


def _format_quiz_history_body(quiz_history_list: list[str] | None) -> str:
    """產出 USER_PROMPT_* 占位 {quiz_history_body}（標題 `## 已出過題目` 在模板內）。"""
    items = _normalize_quiz_history_list(quiz_history_list)
    if not items:
        return _empty_quiz_history_body()
    lines = "\n".join(f"{i}. {q}" for i, q in enumerate(items, start=1))
    return textwrap.dedent(f"""
        **重要：下列題目已經出過，請勿重複出題。** 你本次產生的 `quiz_content` 不得與任一下列題幹相同、近乎相同，或僅做輕微改寫（換句話說、調整選項順序、替換同義詞等）。請改變考查重點、情境或問法，出**一道全新的題目**。

        已出過題目列表：

        {lines}
        """).strip()


def _empty_quiz_history_body() -> str:
    return "（本次請求未提供已出過題目列表；請忽略本節之勿重複限制，依出題 user prompt 與課程內容出題。）"


def _quiz_history_body_for_prompt(
    *,
    quiz_history_list_prompt_text: str = "",
    quiz_history_list: list[str] | None = None,
) -> str:
    """優先使用 API 傳入之 quiz_history_list_prompt_text；否則由題幹陣列格式化（Exam 等相容）。"""
    pt = (quiz_history_list_prompt_text or "").strip()
    if pt:
        return pt
    return _format_quiz_history_body(quiz_history_list)


def _format_quiz_user_message(
    template: str,
    *,
    quiz_user_prompt_text: str = "",
    context_md: str,
    quiz_history_list: list[str] | None = None,
    quiz_history_list_prompt_text: str = "",
) -> str:
    """組 user 訊息：出題 user prompt（獨立區塊）→ 已出過題目（獨立區塊，可空）→ 課程內容。"""
    return template.format(
        quiz_user_prompt_text=(quiz_user_prompt_text or "").strip(),
        quiz_history_body=_quiz_history_body_for_prompt(
            quiz_history_list_prompt_text=quiz_history_list_prompt_text,
            quiz_history_list=quiz_history_list,
        ),
        context_md=context_md,
    )


def _normalize_quiz_history_qa_list(
    quiz_history_list: list[dict[str, str]] | None,
) -> list[tuple[str, str, str, str]]:
    """去除空白項；每項為 (題幹, 作答, 參考答案, 評閱)。"""
    if not quiz_history_list:
        return []
    out: list[tuple[str, str, str, str]] = []
    for item in quiz_history_list:
        if not isinstance(item, dict):
            continue
        q = (item.get("quiz_content") or item.get("question") or "").strip()
        a = (
            item.get("answer_content")
            or item.get("quiz_answer")
            or item.get("answer")
            or ""
        ).strip()
        ref = (
            item.get("quiz_answer_reference")
            or item.get("quiz_reference_answer")
            or item.get("reference_answer")
            or ""
        ).strip()
        critique = (item.get("answer_critique") or item.get("critique") or "").strip()
        raw_comments = item.get("quiz_comments")
        if not critique and raw_comments is not None:
            if isinstance(raw_comments, list):
                critique = "\n".join(str(x).strip() for x in raw_comments if str(x).strip())
            else:
                critique = str(raw_comments).strip()
        if q or a or ref or critique:
            out.append((q, a, ref, critique))
    return out


def _format_quiz_history_qa_body(quiz_history_list: list[dict[str, str]] | None) -> str:
    """產出 USER_PROMPT_COURSE_FOLLOWUP 占位 {quiz_history_qa_body}。"""
    pairs = _normalize_quiz_history_qa_list(quiz_history_list)
    if not pairs:
        return _empty_quiz_history_qa_body()
    blocks: list[str] = []
    for i, (q, a, ref, critique) in enumerate(pairs, start=1):
        blocks.append(
            f"### 第 {i} 組\n\n"
            f"**題目：**\n\n{q or '（未提供題幹）'}\n\n"
            f"**作答：**\n\n{a or '（未提供作答）'}\n\n"
            f"**參考答案：**\n\n{ref or '（未提供）'}\n\n"
            f"**評閱：**\n\n{critique or '（未提供）'}"
        )
    joined = "\n\n---\n\n".join(blocks)
    return textwrap.dedent(f"""
        **重要：** 下列為已完成的問答（每組含題目、作答、參考答案、評閱），為你**接續出題**的依據。區塊內「題目／作答」等標題**僅供閱讀紀錄**，
        你產出的 `quiz_content` **不得仿效**此格式，須**直接寫一道題幹**（無「題目」標籤、無出題前言、無多問列點）。
        **出題方向**（是否針對作答不佳追問、加深、換點等）一律依 system 之「使用者本次出題要求」與「出題 user prompt」決定，**勿**自行依作答好壞判斷方向。
        無論哪種，`quiz_content` **不得**與任一所列題目相同或僅輕微改寫。

        {joined}
        """).strip()


def _empty_quiz_history_qa_body() -> str:
    return (
        "（本次請求未提供先前問答紀錄；請依出題 user prompt 與課程內容出題，"
        "但仍須產出一道完整新題。）"
    )


def format_quiz_history_prompt_for_llm(
    items: list[dict[str, Any]] | None,
    *,
    followup: bool,
) -> str:
    """將 quiz_history_list_prompt_text 物件陣列格式化為 LLM prompt 正文。"""
    if followup:
        return _format_quiz_history_qa_body(items)
    stems = [
        (item.get("quiz_content") or "").strip()
        for item in (items or [])
        if isinstance(item, dict) and (item.get("quiz_content") or "").strip()
    ]
    return _format_quiz_history_body(stems)


def _quiz_history_qa_body_for_prompt(
    *,
    quiz_history_list_prompt_text: str = "",
    quiz_history_list: list[dict[str, str]] | None = None,
) -> str:
    """優先使用 API 傳入之 quiz_history_list_prompt_text；否則由物件陣列格式化（Exam 等相容）。"""
    pt = (quiz_history_list_prompt_text or "").strip()
    if pt:
        return pt
    return _format_quiz_history_qa_body(quiz_history_list)


def _format_quiz_followup_user_message(
    *,
    quiz_user_prompt_text: str = "",
    context_md: str,
    quiz_history_list: list[dict[str, str]] | None = None,
    quiz_history_list_prompt_text: str = "",
) -> str:
    return USER_PROMPT_COURSE_FOLLOWUP.format(
        quiz_user_prompt_text=(quiz_user_prompt_text or "").strip(),
        quiz_history_qa_body=_quiz_history_qa_body_for_prompt(
            quiz_history_list_prompt_text=quiz_history_list_prompt_text,
            quiz_history_list=quiz_history_list,
        ),
        context_md=context_md,
    )


def _context_as_markdown_fenced(context_text: str) -> str:
    """產出 USER_PROMPT_* 占位 {context_md}：Markdown fenced block（標記為 text）；圍欄長度避開內文反引號。"""
    inner = (context_text or "").rstrip()
    max_run = 0
    for m in re.finditer(r"`+", inner):
        max_run = max(max_run, len(m.group(0)))
    n = max(3, max_run + 1)
    fence = "`" * n
    return f"{fence}text\n{inner}\n{fence}"


def _normalize_quiz_llm_json(data: dict) -> dict:
    """
    將 LLM 偶發的別名鍵對齊為資料庫欄位鍵。

    主鍵：`quiz_answer_reference`。相容舊別名：`quiz_reference_answer`、`reference_answer`、`answer`；
    `quiz_hint` 相容 `hint`。
    """
    if "quiz_answer_reference" not in data:
        if "quiz_reference_answer" in data:
            data["quiz_answer_reference"] = data.pop("quiz_reference_answer")
        elif "reference_answer" in data:
            data["quiz_answer_reference"] = data.pop("reference_answer")
        elif "answer" in data:
            data["quiz_answer_reference"] = data.pop("answer")
    if "quiz_hint" not in data and "hint" in data:
        data["quiz_hint"] = data.pop("hint")
    return data


def _invoke_quiz_json_llm(client: OpenAI, messages: list, *, llm_model: str | None = None) -> dict:
    """
    呼叫 GPT 並解析 JSON 物件回應。

    - temperature=0.7：出題需要一定變化；批改在 services/answering 用 0.3，兩者勿混用語意。
    - 若 parse 結果非 dict（極少見），當成空 dict 再 normalize，避免呼叫端 KeyError。
    """
    response = client.chat.completions.create(
        model=llm_model or QUIZ_LLM_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        data = {}
    return _normalize_quiz_llm_json(data)


def _generate_quiz_from_context(
    api_key: str,
    context_text: str,
    *,
    quiz_user_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
    quiz_history_list: list[str] | None = None,
    quiz_history_list_prompt_text: str = "",
    llm_model: str | None = None,
) -> dict:
    """FAISS 與逐字稿共用：組 USER_PROMPT_COURSE → 呼叫 LLM。"""
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 api_key")
    if not (context_text or "").strip():
        raise ValueError("請傳入課程內容（檢索片段或逐字稿，必填）")
    context_md = _context_as_markdown_fenced(context_text)
    user_prompt = _format_quiz_user_message(
        USER_PROMPT_COURSE,
        quiz_user_prompt_text=quiz_user_prompt_text,
        context_md=context_md,
        quiz_history_list=quiz_history_list,
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
    )
    client = OpenAI(api_key=api_key)
    return _invoke_quiz_json_llm(
        client,
        [
            {
                "role": "system",
                "content": _compose_quiz_system_prompt(
                    SYSTEM_PROMPT_QUIZ, quiz_system_prompt_text=quiz_system_prompt_text
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        llm_model=llm_model,
    )


def _generate_quiz_followup_from_context(
    api_key: str,
    context_text: str,
    *,
    quiz_user_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
    quiz_history_list: list[dict[str, str]] | None = None,
    quiz_history_list_prompt_text: str = "",
    llm_model: str | None = None,
) -> dict:
    """追問出題：組 USER_PROMPT_COURSE_FOLLOWUP → 呼叫 LLM。"""
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 api_key")
    if not (context_text or "").strip():
        raise ValueError("請傳入課程內容（檢索片段或逐字稿，必填）")
    context_md = _context_as_markdown_fenced(context_text)
    user_prompt = _format_quiz_followup_user_message(
        quiz_user_prompt_text=quiz_user_prompt_text,
        context_md=context_md,
        quiz_history_list=quiz_history_list,
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
    )
    client = OpenAI(api_key=api_key)
    return _invoke_quiz_json_llm(
        client,
        [
            {
                "role": "system",
                "content": _compose_quiz_system_prompt(
                    SYSTEM_PROMPT_QUIZ_FOLLOWUP, quiz_system_prompt_text=quiz_system_prompt_text
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        llm_model=llm_model,
    )


# ---------------------------------------------------------------------------
# 公開 API（由 routers/exam、routers/answer 動態 import）
# ---------------------------------------------------------------------------

def generate_quiz_transcript_only(
    api_key: str,
    transcript: str,
    quiz_history_list: list[str] | None = None,
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str = "",
    quiz_history_list_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
) -> dict:
    """
    無 FAISS：與 generate_quiz 相同訊息結構——system 為出題規範；逐字稿置於 user「課程內容」。

    路由：Rag_Unit.unit_type 為 2／3／4 時使用（與 generate_quiz 互斥）。
    回傳：quiz_content, quiz_hint, quiz_answer_reference。

    Args:
        api_key: OpenAI API Key（與 embeddings 無關，本路徑不建向量）。
        transcript: 課程全文或逐字稿，填入 user 課程內容區塊；不可空。
        quiz_user_prompt_text: 填入 USER_PROMPT_* 之「出題 user prompt」占位；空字串時依 system 指示略過該節。
        quiz_history_list: 已出過題目題幹（Exam 等相容；Rag 請改傳 quiz_history_list_prompt_text）。
        quiz_history_list_prompt_text: 併入 prompt 的先前問答正文（Rag llm-generate 使用）。
    """
    raw_tc = transcript if transcript is not None else ""
    if not raw_tc.strip():
        raise ValueError("請傳入 transcript（課程內容區塊，必填）")
    return _generate_quiz_from_context(
        api_key,
        raw_tc,
        quiz_user_prompt_text=quiz_user_prompt_text,
        quiz_system_prompt_text=quiz_system_prompt_text,
        quiz_history_list=quiz_history_list,
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
        llm_model=llm_model,
    )


def generate_quiz(
    zip_path: Path,
    api_key: str,
    quiz_history_list: list[str] | None = None,
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str = "",
    quiz_history_list_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
) -> dict:
    """
    有 FAISS RAG ZIP：解壓 → 載入向量庫 → 檢索 → 組 Markdown user → LLM。

    zip_path 須為 POST /v1/rag/pages/{rag_page_id}/build-zip 產物。prompt 與逐字稿路徑共用 USER_PROMPT_COURSE。
    回傳：quiz_content, quiz_hint, quiz_answer_reference。

    Args:
        zip_path: 本機路徑，指向已下載之 RAG ZIP（內含 index.faiss / index.pkl）。
        api_key: 同時用於 OpenAIEmbeddings 與 Chat Completions。
        quiz_user_prompt_text: 填入 USER_PROMPT_COURSE 之「出題 user prompt」占位；空字串時依 system 指示略過該節。
        quiz_history_list: 已出過題目題幹（Exam 等相容；Rag 請改傳 quiz_history_list_prompt_text）。
        quiz_history_list_prompt_text: 併入 prompt 的先前問答正文（Rag llm-generate 使用）。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 api_key")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        # Python 3.11+：ZIP 內 UTF-8 檔名需 metadata_encoding，否則解壓可能觸發 ascii 編碼錯誤。
        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        # 僅接受 LangChain FAISS.save_local 目錄結構，與自建講義 ZIP 區隔。
        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError(
                "此 API 僅支援 RAG ZIP（由 POST /v1/rag/pages/{rag_page_id}/build-zip 產出），請上傳含 FAISS 向量庫的 ZIP"
            )

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        # 本機自建 ZIP：須允許 pickle 反序列化（LangChain 預設關閉）。
        vectorstore = FAISS.load_local(
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        docs = retriever.invoke(DEFAULT_RETRIEVAL_QUERY)
        # page_content：每段 chunk 純文字；多段之間空行分隔，便於模型閱讀邊界。
        context_text = "\n\n".join(d.page_content for d in docs)
        return _generate_quiz_from_context(
            api_key,
            context_text,
            quiz_user_prompt_text=quiz_user_prompt_text,
            quiz_system_prompt_text=quiz_system_prompt_text,
            quiz_history_list=quiz_history_list,
            quiz_history_list_prompt_text=quiz_history_list_prompt_text,
            llm_model=llm_model,
        )
    finally:
        # 暫存目錄必清，避免磁碟堆積與路徑外洩。
        shutil.rmtree(extract_folder, ignore_errors=True)


def generate_quiz_followup_transcript_only(
    api_key: str,
    transcript: str,
    quiz_history_list: list[dict[str, str]] | None = None,
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str = "",
    quiz_history_list_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
) -> dict:
    """
    追問出題（無 FAISS）：答不好追問弱點，答好出新題；quiz_history_list 為先前問答（題幹＋作答）列表。
    """
    raw_tc = transcript if transcript is not None else ""
    if not raw_tc.strip():
        raise ValueError("請傳入 transcript（課程內容區塊，必填）")
    return _generate_quiz_followup_from_context(
        api_key,
        raw_tc,
        quiz_user_prompt_text=quiz_user_prompt_text,
        quiz_system_prompt_text=quiz_system_prompt_text,
        quiz_history_list=quiz_history_list,
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
        llm_model=llm_model,
    )


def generate_quiz_followup(
    zip_path: Path,
    api_key: str,
    quiz_history_list: list[dict[str, str]] | None = None,
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str = "",
    quiz_history_list_prompt_text: str = "",
    quiz_system_prompt_text: str = "",
) -> dict:
    """追問出題（有 FAISS RAG ZIP）：答不好追問弱點，答好出新題。"""
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 api_key")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError(
                "此 API 僅支援 RAG ZIP（由 POST /v1/rag/pages/{rag_page_id}/build-zip 產出），請上傳含 FAISS 向量庫的 ZIP"
            )

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        vectorstore = FAISS.load_local(
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        docs = retriever.invoke(DEFAULT_RETRIEVAL_QUERY)
        context_text = "\n\n".join(d.page_content for d in docs)
        return _generate_quiz_followup_from_context(
            api_key,
            context_text,
            quiz_user_prompt_text=quiz_user_prompt_text,
            quiz_system_prompt_text=quiz_system_prompt_text,
            quiz_history_list=quiz_history_list,
            quiz_history_list_prompt_text=quiz_history_list_prompt_text,
            llm_model=llm_model,
        )
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)
