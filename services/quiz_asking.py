"""
Quiz（測驗／Test）追問回答管線 service（POST /v1/quiz/groups/{quiz_group_id}/llm-ask）。

學生在題組出題後，對該題組對應 **Bank 單元的課程內容**仍有不懂之處時發問；
本模組以 bank 的課程內容（逐字稿／FAISS 向量檢索片段）回答學生問題。
回傳純 Markdown 答案字串（**非** JSON）。

**quiz 專屬**：自 services/asking.py（exam/rag 管線）的設計仿作，但完全獨立、僅依賴 bank 管線
（services.bank_answering 的 embedding／檢索常數、utils.bank_faiss、utils.llm_error），
與 exam／rag 程式無共用。同步執行；呼叫端取得答案字串後 INSERT 一列 public."Quiz_Ask"。

追問定位在**題組**（非單題）：prompt 帶題組之出題 user prompt（教學脈絡）、本題組全部測驗題紀錄
（題目／提示／參考答案／學生作答／評閱）、先前追問紀錄，以及本次學生提問，依課程內容作答。
"""

import logging
import os
import textwrap
import zipfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from services.bank_answering import (
    BANK_ANSWER_EMBEDDING_MODEL,
    BANK_ANSWER_LLM_MODEL,
    BANK_ANSWER_RAG_CHUNK_OVERLAP,
    BANK_ANSWER_RAG_CHUNK_SIZE,
    BANK_ANSWER_RETRIEVAL_K,
    _answer_field_display,
)
from services.bank_generation import context_as_markdown_fenced
from utils.llm_error import LlmCallError, format_llm_error, is_llm_call_error
from utils.bank_faiss import process_zip_to_docs

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 追問回答 Prompt（quiz 專屬，可自由修改）
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_QUIZ_ASK = textwrap.dedent("""
    # 角色

    你是一位授課教師／助教。學生在本題組測驗中，對於課程內容仍有不懂之處，因而發問。
    請依**課程內容**（逐字稿／檢索片段）、**題組測驗紀錄**與**先前追問紀錄**耐心回答學生的問題，協助其理解。

    ## 指令優先級（必須遵守）

    - 使用者訊息中的 **`## 出題 user prompt`** 為教師設定的**教學脈絡**，優先級**高於**「回答說明」等泛化規則。
    - 該節內文有**實質要求**時（非僅空白或占位句如「（未提供）」），請據以理解命題焦點與教學重點。
    - **`## 題組測驗紀錄`**、**`## 先前追問紀錄`** 為本題組脈絡；回答時應與測驗題、學生作答與先前追問**銜接一致**，勿與紀錄矛盾。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。
    - **`## 題組測驗紀錄`**：本題組各題之題目、提示、參考答案、學生作答、評閱（後三者可能為空）。
    - **`## 先前追問紀錄`**：本題組內**已完成**之追問與回答（不含本次 `## 學生提問`）。
    - **`## 學生提問`**：本次待回答之問題，為你**首要回應對象**。

    ## 回答原則

    - **僅能**依據**課程內容**所提供之引用作答；課程內容未涵蓋之處，請明確說明「課程內容未提及」，**勿杜撰**。
    - 緊扣**本次**學生提問（`## 學生提問`）；可參考測驗紀錄與先前追問脈絡，但勿偏離本次問題。
    - 學生指涉「剛才」「上一題」「前面問的」等時，請對照 **先前追問紀錄** 與 **題組測驗紀錄** 釐清所指。
    - 直接回傳**一段 Markdown 純文字**作為答覆（**勿**包成 JSON、**勿**輸出鍵值物件）。
    """).strip()

USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE = textwrap.dedent("""
    {id_block}## 出題 user prompt

    {question_user_prompt_text}

    ## 題組測驗紀錄

    {quiz_qa_history_body}

    ## 先前追問紀錄

    {ask_history_body}

    ## 學生提問

    {ask_user_prompt_text}

    ## 回答說明

    - 緊扣**本次學生提問**，依**課程內容**作答；可參考**題組測驗紀錄**與**先前追問紀錄**銜接脈絡。
    - 課程內容未涵蓋者請明說「課程內容未提及」，勿杜撰。
    - 回傳**一段 Markdown 純文字**；**勿**包成 JSON 或鍵值物件。

    ---

    ## 課程內容

    下列為課程逐字稿／全文（回答**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()

USER_PROMPT_QUIZ_ASK_FAISS_COURSE = textwrap.dedent("""
    {id_block}## 出題 user prompt

    {question_user_prompt_text}

    ## 題組測驗紀錄

    {quiz_qa_history_body}

    ## 先前追問紀錄

    {ask_history_body}

    ## 學生提問

    {ask_user_prompt_text}

    ## 回答說明

    - 緊扣**本次學生提問**，依**課程內容**作答；可參考**題組測驗紀錄**與**先前追問紀錄**銜接脈絡。
    - 課程內容未涵蓋者請明說「課程內容未提及」，勿杜撰。
    - 回傳**一段 Markdown 純文字**；**勿**包成 JSON 或鍵值物件。

    ---

    ## 課程內容

    下列為自課程向量庫檢索之片段（回答**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()


def _history_field_display(raw: str, *, empty_label: str = "（未提供）") -> str:
    return (raw or "").strip() or empty_label


def format_quiz_qa_history_body(qa_rows: list[dict] | None) -> str:
    """將本題組 Quiz_QA 列格式化成 prompt 正文（題目／提示／參考答案／學生作答／評閱）。"""
    if not qa_rows:
        return "（本題組尚無測驗題目紀錄。）"
    blocks: list[str] = []
    for i, row in enumerate(qa_rows, start=1):
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("question_series_index") or 0)
        except (TypeError, ValueError):
            idx = 0
        label = idx if idx > 0 else i
        blocks.append(
            f"### 第 {label} 題\n\n"
            f"**題目：**\n\n{_history_field_display(row.get('question_content'))}\n\n"
            f"**提示：**\n\n{_history_field_display(row.get('question_hint'))}\n\n"
            f"**參考答案：**\n\n{_history_field_display(row.get('question_answer_reference'))}\n\n"
            f"**學生作答：**\n\n{_history_field_display(row.get('answer_content'))}\n\n"
            f"**評閱：**\n\n{_history_field_display(row.get('answer_critique'))}"
        )
    if not blocks:
        return "（本題組尚無測驗題目紀錄。）"
    return "\n\n---\n\n".join(blocks)


def format_ask_history_body(ask_rows: list[dict] | None) -> str:
    """將本題組已完成之 Quiz_Ask 列格式化成 prompt 正文（提問／回答）。"""
    if not ask_rows:
        return "（尚無先前追問紀錄。）"
    blocks: list[str] = []
    for i, row in enumerate(ask_rows, start=1):
        if not isinstance(row, dict):
            continue
        ask_text = (row.get("ask_user_prompt_text") or "").strip()
        answer_text = (row.get("answer_content") or "").strip()
        if not ask_text and not answer_text:
            continue
        blocks.append(
            f"### 第 {i} 次追問\n\n"
            f"**提問：**\n\n{_history_field_display(ask_text)}\n\n"
            f"**回答：**\n\n{_history_field_display(answer_text)}"
        )
    if not blocks:
        return "（尚無先前追問紀錄。）"
    return "\n\n---\n\n".join(blocks)


def _quiz_ask_id_block(quiz_group_id: int | None, bank_group_id: int | None) -> str:
    """關聯識別 Markdown（格式與 bank_answering 之 id_block 一致）。"""
    id_lines: list[str] = []
    if quiz_group_id is not None and quiz_group_id > 0:
        id_lines.append(f"- **quiz_group_id**：`{quiz_group_id}`")
    if bank_group_id is not None and bank_group_id > 0:
        id_lines.append(f"- **bank_group_id**：`{bank_group_id}`")
    return ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""


def _quiz_ask_llm(api_key: str, user_msg: str, llm_model: str | None) -> str:
    """以 SYSTEM_PROMPT_QUIZ_ASK + user 訊息呼叫 LLM，回傳答案純文字。"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=llm_model or BANK_ANSWER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QUIZ_ASK},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
    except Exception as e:
        if is_llm_call_error(e):
            raise LlmCallError(format_llm_error(e)) from e
        raise
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# 逐字稿純 LLM 回答（unit_type 2／3／4，不讀 RAG ZIP）
# ---------------------------------------------------------------------------

def run_quiz_ask_transcript_only(
    api_key: str,
    transcript: str,
    ask_user_prompt_text: str = "",
    *,
    question_user_prompt_text: str = "",
    quiz_qa_history_body: str = "",
    ask_history_body: str = "",
    quiz_group_id: int | None = None,
    bank_group_id: int | None = None,
    llm_model: str | None = None,
) -> str:
    """無 RAG ZIP（unit_type 2／3／4）：system=SYSTEM_PROMPT_QUIZ_ASK；user=逐字稿模板。回傳答案純文字。"""
    ts = (transcript or "").strip()
    if not ts:
        raise ValueError("回答用 transcript 未設定")
    user_msg = USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE.format(
        id_block=_quiz_ask_id_block(quiz_group_id, bank_group_id),
        question_user_prompt_text=_answer_field_display(question_user_prompt_text),
        quiz_qa_history_body=(quiz_qa_history_body or "").strip() or format_quiz_qa_history_body(None),
        ask_history_body=(ask_history_body or "").strip() or format_ask_history_body(None),
        ask_user_prompt_text=_answer_field_display(ask_user_prompt_text),
        context_md=context_as_markdown_fenced(ts),
    )
    return _quiz_ask_llm(api_key, user_msg, llm_model)


# ---------------------------------------------------------------------------
# RAG ZIP 回答（FAISS 或自講義建臨時向量庫）
# ---------------------------------------------------------------------------

def run_quiz_ask_job(
    work_dir: Path,
    api_key: str,
    ask_user_prompt_text: str = "",
    *,
    question_user_prompt_text: str = "",
    quiz_qa_history_body: str = "",
    ask_history_body: str = "",
    quiz_group_id: int | None = None,
    bank_group_id: int | None = None,
    unit_type: int = 0,
    llm_model: str | None = None,
) -> str:
    """
    在給定的 work_dir（已含 ref.zip）執行向量檢索 + LLM 回答。回傳答案純文字。

    work_dir：由呼叫端建立；內含 ref.zip（bank 的 RAG 或講義壓縮檔）與子目錄 extract（解壓目標）。
    檢索查詢即學生提問文字。
    """
    zip_source_path = work_dir / "ref.zip"
    extract_folder = work_dir / "extract"
    extract_folder.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(zip_source_path):
        raise ValueError("無效的 ZIP 檔")

    with zipfile.ZipFile(zip_source_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    # 有 FAISS 檔則載入既有向量庫；否則自 ZIP 講義建臨時向量庫（路徑與 unit_type 須一致）。
    is_rag_db = False
    db_folder = None
    for root, _, files in os.walk(extract_folder):
        if "index.faiss" in files and "index.pkl" in files:
            is_rag_db = True
            db_folder = root
            break

    embeddings = OpenAIEmbeddings(model=BANK_ANSWER_EMBEDDING_MODEL, api_key=api_key)

    if is_rag_db:
        vectorstore = FAISS.load_local(
            db_folder,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        all_documents = process_zip_to_docs(zip_source_path, extract_folder, unit_type=unit_type)
        if not all_documents:
            raise ValueError("ZIP 內無支援的講義文件（請確認單元 unit_type 與檔案格式一致）")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=BANK_ANSWER_RAG_CHUNK_SIZE,
            chunk_overlap=BANK_ANSWER_RAG_CHUNK_OVERLAP,
        )
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": BANK_ANSWER_RETRIEVAL_K})
    docs = retriever.invoke((ask_user_prompt_text or "").strip())
    context_text = "\n\n".join([d.page_content for d in docs])
    context_md = context_as_markdown_fenced(context_text)

    user_msg = USER_PROMPT_QUIZ_ASK_FAISS_COURSE.format(
        id_block=_quiz_ask_id_block(quiz_group_id, bank_group_id),
        question_user_prompt_text=_answer_field_display(question_user_prompt_text),
        quiz_qa_history_body=(quiz_qa_history_body or "").strip() or format_quiz_qa_history_body(None),
        ask_history_body=(ask_history_body or "").strip() or format_ask_history_body(None),
        ask_user_prompt_text=_answer_field_display(ask_user_prompt_text),
        context_md=context_md,
    )
    return _quiz_ask_llm(api_key, user_msg, llm_model)
