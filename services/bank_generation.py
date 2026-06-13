"""
Bank（測試題庫）題目 LLM 生成 — **bank 專屬，與 rag／exam 完全無關**（prompt 與管線皆獨立，可自由修改不影響其他模組）。

- generate_bank_quiz：有 FAISS RAG ZIP（unit_type=1）→ 向量檢索片段為課程內容。
- generate_bank_quiz_transcript_only：unit_type 2／3／4 → 逐字稿為課程內容。
- 兩路徑共用同一 system／user prompt 與 `_generate_bank_quiz_from_context`。
- 無「追問」概念；連續出題以 question_system_prompt_text（最高優先）＋已出過題目（勿重複）達成。

embedding 模型與 RETRIEVAL 設定須與 build-zip 建索引時一致，否則向量維度不相容。
"""

import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# -----------------------------------------------------------------------------
# 模型與檢索常數（bank 專屬；embedding 須與 build-zip 一致）
# -----------------------------------------------------------------------------
BANK_QUIZ_LLM_MODEL = "gpt-5.4"
BANK_EMBEDDING_MODEL = "text-embedding-3-small"
BANK_RETRIEVAL_K = 5
BANK_DEFAULT_RETRIEVAL_QUERY = "課程重點概念"


# -----------------------------------------------------------------------------
# LLM 出題 Prompt（bank 專屬，可自由修改）
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_BANK_QUIZ = textwrap.dedent("""
    # 角色

    你是一位教授，請為學生設計測驗題目。

    ## 指令優先級（必須遵守）

    - 使用者訊息中 **`## 出題 user prompt`** 以下之內文為教師下給你的**直接出題指令**，優先級**高於**本 system 之泛化規則與 **課程內容** 之呈現方式。
    - 該節有**實質文字**時（非僅空白或占位），**必須完整遵守**（題型、難度、焦點、格式、用語等），不得因課程片段較易取材而偏離該節要求。
    - 僅當該節無實質文字時，始依 **課程內容** 與本訊息其餘規範出題。{quiz_system_prompt_text}
    - 使用者訊息中的 **`## 追問紀錄`** 為本題組學生對課程內容的追問與回答，可作為**接續出題**之參考（例如針對學生困惑之處出題）；**勿**因此重複 system 出題歷史中已出過之題幹。

    ## 出題歷史（接續出題依據，必須遵守）

    下列為本題組**已出過的題目**（出題歷史），為**權威依據**：

    {quiz_history_body}

    - 請在上述出題歷史的基礎上，**接續出下一題**：延續本題組的主題與脈絡，並依「指令優先級」之使用者本次出題要求（若有）決定遞進方向（例如越來越深入、換考查點）。
    - **不得重複**：出題前**逐題比對**上述歷史，你產生的新題不可與其中任一題相同、近乎相同，或僅改寫措辭／調整順序／替換同義詞；須改變考查點、情境或問法，使學生能辨識為**另一道新題**。
    - 若上方標示未提供出題歷史，則視為本題組第一題，直接依出題 user prompt 與課程內容出題。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。
    - **出題 user prompt** 區塊（`## 出題 user prompt`）以下若無實質文字（僅留白或空字串），請**完全忽略**該節，僅依 **課程內容** 與上方出題歷史出題；有文字時須**先滿足該節指令**，再於課程引用範圍內取材。
    - 出題歷史（已出過題目）見本 system 之 **`## 出題歷史`** 區塊，**不在**使用者訊息中。

    ## 題數限制（必須遵守）

    - 每次任務**僅出一道題**；JSON 中 `quiz_content` **只能包含一個題幹**（一個明確的發問或作答要求）。
    - **禁止**一次出多題：勿使用「第 1 題／第 2 題」、編號題組、多選題卷、小測驗卷、題目列表等格式。
    - 若教師指令或課程內容暗示多題，仍須**只選一個**最符合要求的考查點出成**單一題目**。
    - `quiz_hint` 與 `quiz_answer_reference` 僅能對應**這一道題**，勿涵蓋多題答案。

    ## 題幹寫法（必須遵守）

    - `quiz_content` **只寫給學生看的題目本身**：開頭即為發問或明確作答任務，學生應能**直接作答**，無需再讀出題說明或試卷前言。
    - **禁止**在 `quiz_content` 出現：
      - 版面標籤：「題目」「中文題目：」「Question:」等
      - 引導套話：「根據課程／上文／研究…」「請回答以下問題」「回答下列問題」
      - 列點、編號的**多個子問題**（僅允許**一個**問句或一道作答任務）
      - 作答／評分指示：「請確保你的答案…」「請詳述並舉例…」（必要時放 `quiz_hint`，勿放題幹）
    - **正例**（`quiz_content`）：「研究中，指導參與者進行夢境引導時，研究人員可能遇到哪些困難？」
    - **反例**：先寫「中文題目：…」＋「根據課程…請回答以下問題」再列 (1)(2) 多問。

    ## 出題規範

    - `quiz_content`、`quiz_hint`、`quiz_answer_reference` 之字串值皆為 **Markdown**（段落、清單、`**強調**` 等）；用語與語種請依 **出題 user prompt**（有實質文字時）與課程內容。
    - 依上方 **出題歷史** 接續出題時：`quiz_content` **不得**重複或實質重複歷史中任一題；`quiz_hint` 與 `quiz_answer_reference` 亦須對應這道**新題**，勿沿用舊題答案結構敷衍。

    ## 出題程序與回傳格式（JSON）

    **理由先行**：請**先**綜合「出題歷史、追問紀錄、使用者本次出題要求（題組規則／連續出題規定）、出題 user prompt（出題規則）與課程內容」決定本題的**出題理由**，**再**依該理由寫出題目與答案；`question_reason` 須與最終題目一致。

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `question_reason`：**出題理由**（Markdown 字串）：說明本題要考查的重點概念／能力與為何此時出此題；**若有**「使用者本次出題要求（題組規則／連續出題規定）」或「出題 user prompt（出題規則）」，須說明本題**如何呼應／落實**之；並說明與**出題歷史**的延續／遞進關係，以及是否回應**追問紀錄**中的學生困惑（如有）。若無相關規則或歷史，僅就題目本身說明。
    - `quiz_content`：**單一**題目題幹（Markdown 字串；**直接寫題，勿加「題目」等標籤或出題前言**），須與 `question_reason` 一致。
    - `quiz_hint`：該題答案提示（Markdown 字串）
    - `quiz_answer_reference`：該題參考答案（Markdown 字串）
    """).strip()


def _compose_bank_quiz_system_prompt(
    base_system: str, *, quiz_system_prompt_text: str, quiz_history_body: str = ""
) -> str:
    """將題組 question_system_prompt_text 織入「指令優先級」、出題歷史織入「出題歷史」區塊；皆於 system。"""
    extra = (quiz_system_prompt_text or "").strip()
    user_requirement = (
        f"\n- **使用者本次出題要求（最高優先，必須遵守）**：{extra}" if extra else ""
    )
    body = (quiz_history_body or "").strip() or "（本次未提供出題歷史；視為本題組第一題。）"
    return base_system.format(quiz_system_prompt_text=user_requirement, quiz_history_body=body)


USER_PROMPT_BANK_COURSE = textwrap.dedent("""
    ## 必須遵守（最高優先）

    下節 **出題 user prompt** 為教師直接指令，必須一定遵守。
    **出題歷史（已出過題目）見 system 之「## 出題歷史」**；請在該歷史基礎上**接續出下一題**，且不與其中任一題重複。
    若有 **追問紀錄**，可參考學生困惑之處調整本題考查重點，但仍須遵守出題歷史之勿重複規定。
    **`quiz_content` 請直接寫題幹**（學生可立即作答的一句／一段發問），勿加「題目」標題、「根據課程…」前言或多問列點。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 追問紀錄

    {ask_history_body}

    ---

    ## 課程內容

    下列為課程內容（向量檢索片段或完整逐字稿；出題**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()


def _empty_ask_history_body() -> str:
    return "（尚無追問紀錄；請忽略本節。）"


# ---------------------------------------------------------------------------
# 已出過題目區塊
# ---------------------------------------------------------------------------

def _normalize_stem_list(stems: list[str] | None) -> list[str]:
    if not stems:
        return []
    return [s.strip() for s in stems if (s or "").strip()]


def _format_bank_quiz_history_body(stems: list[str] | None) -> str:
    items = _normalize_stem_list(stems)
    if not items:
        return "（本次請求未提供已出過題目列表；請忽略本節之勿重複限制，依出題 user prompt 與課程內容出題。）"
    lines = "\n".join(f"{i}. {q}" for i, q in enumerate(items, start=1))
    return textwrap.dedent(f"""
        **重要：下列題目已經出過，請勿重複出題。** 你本次產生的 `quiz_content` 不得與任一下列題幹相同、近乎相同，或僅做輕微改寫（換句話說、調整選項順序、替換同義詞等）。請改變考查重點、情境或問法，出**一道全新的題目**。

        已出過題目列表：

        {lines}
        """).strip()


def format_bank_quiz_history_prompt_for_llm(items: list[dict[str, Any]] | None) -> str:
    """將「已出過題目」物件陣列（每筆含 quiz_content）格式化為 LLM prompt 正文。"""
    stems = [
        (item.get("quiz_content") or "").strip()
        for item in (items or [])
        if isinstance(item, dict) and (item.get("quiz_content") or "").strip()
    ]
    return _format_bank_quiz_history_body(stems)


def _quiz_history_body_for_prompt(*, quiz_history_list_prompt_text: str, quiz_history_list: list[str] | None = None) -> str:
    pt = (quiz_history_list_prompt_text or "").strip()
    if pt:
        return pt
    return _format_bank_quiz_history_body(quiz_history_list)


def _format_bank_quiz_user_message(
    *,
    quiz_user_prompt_text: str,
    context_md: str,
    ask_history_body: str = "",
) -> str:
    return USER_PROMPT_BANK_COURSE.format(
        quiz_user_prompt_text=(quiz_user_prompt_text or "").strip(),
        ask_history_body=(ask_history_body or "").strip() or _empty_ask_history_body(),
        context_md=context_md,
    )


# ---------------------------------------------------------------------------
# 課程內容 fenced、LLM 回傳正規化與呼叫
# ---------------------------------------------------------------------------

def context_as_markdown_fenced(context_text: str) -> str:
    """產出 {context_md}：Markdown fenced block（標記 text）；圍欄長度避開內文反引號。"""
    inner = (context_text or "").rstrip()
    max_run = 0
    for m in re.finditer(r"`+", inner):
        max_run = max(max_run, len(m.group(0)))
    n = max(3, max_run + 1)
    fence = "`" * n
    return f"{fence}text\n{inner}\n{fence}"


def _normalize_quiz_llm_json(data: dict) -> dict:
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
    response = client.chat.completions.create(
        model=llm_model or BANK_QUIZ_LLM_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        data = {}
    return _normalize_quiz_llm_json(data)


def _generate_bank_quiz_from_context(
    api_key: str,
    context_text: str,
    *,
    quiz_user_prompt_text: str,
    quiz_system_prompt_text: str,
    quiz_history_list: list[str] | None = None,
    quiz_history_list_prompt_text: str,
    ask_history_body: str = "",
    llm_model: str | None = None,
) -> dict:
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 api_key")
    if not (context_text or "").strip():
        raise ValueError("請傳入課程內容（檢索片段或逐字稿，必填）")
    context_md = context_as_markdown_fenced(context_text)
    history_body = _quiz_history_body_for_prompt(
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
        quiz_history_list=quiz_history_list,
    )
    system_content = _compose_bank_quiz_system_prompt(
        SYSTEM_PROMPT_BANK_QUIZ,
        quiz_system_prompt_text=quiz_system_prompt_text,
        quiz_history_body=history_body,
    )
    user_prompt = _format_bank_quiz_user_message(
        quiz_user_prompt_text=quiz_user_prompt_text,
        context_md=context_md,
        ask_history_body=ask_history_body,
    )
    client = OpenAI(api_key=api_key)
    return _invoke_quiz_json_llm(
        client,
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
        llm_model=llm_model,
    )


# ---------------------------------------------------------------------------
# 公開 API（由 routers/bank 呼叫）
# ---------------------------------------------------------------------------

def generate_bank_quiz_transcript_only(
    api_key: str,
    transcript: str,
    quiz_history_list: list[str] | None = None,
    ask_history_body: str = "",
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str,
    quiz_history_list_prompt_text: str,
    quiz_system_prompt_text: str,
) -> dict:
    """無 FAISS（unit_type 2／3／4）：逐字稿置於 user 課程內容區塊。回傳 quiz_content/quiz_hint/quiz_answer_reference。"""
    raw_tc = transcript if transcript is not None else ""
    if not raw_tc.strip():
        raise ValueError("請傳入 transcript（課程內容區塊，必填）")
    return _generate_bank_quiz_from_context(
        api_key,
        raw_tc,
        quiz_user_prompt_text=quiz_user_prompt_text,
        quiz_system_prompt_text=quiz_system_prompt_text,
        quiz_history_list=quiz_history_list,
        quiz_history_list_prompt_text=quiz_history_list_prompt_text,
        ask_history_body=ask_history_body,
        llm_model=llm_model,
    )


def generate_bank_quiz(
    zip_path: Path,
    api_key: str,
    quiz_history_list: list[str] | None = None,
    ask_history_body: str = "",
    llm_model: str | None = None,
    *,
    quiz_user_prompt_text: str,
    quiz_history_list_prompt_text: str,
    quiz_system_prompt_text: str,
) -> dict:
    """有 FAISS RAG ZIP（unit_type=1）：解壓 → 載入向量庫 → 檢索 → 組 user → LLM。"""
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
                "此 API 僅支援 RAG ZIP（由 POST /v1/bank/pages/{bank_page_id}/build-zip 產出），請上傳含 FAISS 向量庫的 ZIP"
            )

        embeddings = OpenAIEmbeddings(model=BANK_EMBEDDING_MODEL, api_key=api_key)
        vectorstore = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever(search_kwargs={"k": BANK_RETRIEVAL_K})
        docs = retriever.invoke(BANK_DEFAULT_RETRIEVAL_QUERY)
        context_text = "\n\n".join(d.page_content for d in docs)
        return _generate_bank_quiz_from_context(
            api_key,
            context_text,
            quiz_user_prompt_text=quiz_user_prompt_text,
            quiz_system_prompt_text=quiz_system_prompt_text,
            quiz_history_list=quiz_history_list,
            quiz_history_list_prompt_text=quiz_history_list_prompt_text,
            ask_history_body=ask_history_body,
            llm_model=llm_model,
        )
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)
