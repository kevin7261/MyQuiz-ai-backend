"""
追問回答管線 service（POST /v1/exam/quizzes/llm-ask）。

學生於作答某 Exam_Quiz 題目後，對課程內容仍有不懂之處時針對該題追問細節；
本模組以 **RAG 課程內容**（逐字稿／向量檢索片段）回答學生問題，必要時參考題幹、
學生作答與批改評語，針對其困惑點解說。回傳純 Markdown 答案字串（**非**評分 JSON）。

檔案結構（仿 `services/answering.py`）：
1. 追問回答 Prompt（`SYSTEM_PROMPT_ASK`、`USER_PROMPT_ASK_TRANSCRIPT_COURSE`、`USER_PROMPT_ASK_FAISS_COURSE`）
2. 逐字稿純 LLM 回答（unit_type 2／3／4）與 RAG ZIP（FAISS／臨時向量庫）回答

重要：
- 與「批改」`services/answering.py` 共用 embedding 模型、檢索 k、chunk 常數，維持索引維度一致。
- 同步執行（不走背景 job），呼叫端取得答案字串後即 INSERT 一列 public."Exam_Ask"。
"""

import logging
import os
import zipfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from services.answering import (
    ANSWER_EMBEDDING_MODEL,
    ANSWER_LLM_MODEL,
    ANSWER_RAG_CHUNK_OVERLAP,
    ANSWER_RAG_CHUNK_SIZE,
    ANSWER_RETRIEVAL_K,
    _answer_field_display,
)
from services.quiz_generation import _context_as_markdown_fenced
from utils.llm_error import LlmCallError, format_llm_error, is_llm_call_error
from utils.rag_faiss import process_zip_to_docs

import textwrap

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 追問回答 Prompt（分段同 answering.py：`SYSTEM_PROMPT_*` → `USER_PROMPT_*_COURSE`、`---`、`## 課程內容`；user 僅 `.format(...)`）
# -----------------------------------------------------------------------------
# Chat messages：
#   role=system … SYSTEM_PROMPT_ASK
#   role=user … USER_PROMPT_ASK_TRANSCRIPT_COURSE 或 USER_PROMPT_ASK_FAISS_COURSE；
#   其中 `{context_md}` 僅由逐字稿／向量檢索經 `_context_as_markdown_fenced` 產生，其餘占位同一 `.format` 填入。

SYSTEM_PROMPT_ASK = textwrap.dedent("""
    # 角色

    你是一位授課教師／助教。學生在作答本測驗題後，對於課程內容仍有不懂之處，因而針對本題追問細節。
    請依**課程內容**（逐字稿／檢索片段）耐心回答學生的問題，必要時對照題幹、學生作答與批改評語，
    針對其困惑或答錯之處加以說明，協助其理解。

    ## 指令優先級（必須遵守）

    - 使用者訊息中的 **`## 出題 user prompt`** 與 **`## 作答 user prompt`** 為教師下給你的**直接指令**，優先級**高於**題幹、課程引用與「回答說明」等泛化規則。
    - 兩節內文有**實質要求**時（非僅空白或占位句如「（未提供）」），**必須完整遵守**，不得忽略、弱化或改寫其意圖。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。

    ## 回答原則

    - **僅能**依據**課程內容**所提供之引用作答；課程內容未涵蓋之處，請明確說明「課程內容未提及」，**勿杜撰**。
    - 緊扣學生的提問（`## 學生追問`）；若學生係因作答有誤而不解，請對照其作答與批改評語點出觀念落差。
    - 直接回傳**一段 Markdown 純文字**作為答覆（**勿**包成 JSON、**勿**輸出鍵值物件）。
    """).strip()

USER_PROMPT_ASK_TRANSCRIPT_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**回答說明**）牴觸時，**以該兩節為準**。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 作答 user prompt

    {answer_user_prompt_text}

    ## quiz_content（測驗題目）

    {quiz_content}

    ## answer_content（學生先前作答）

    {quiz_answer}

    ## answer_critique（先前批改評語）

    {answer_critique}

    ## 學生追問

    {ask_user_prompt_text}

    ## 回答說明

    - 緊扣**學生追問**，依**課程內容**作答；課程內容未涵蓋者請明說「課程內容未提及」，勿杜撰。
    - 回傳**一段 Markdown 純文字**；**勿**包成 JSON 或鍵值物件。

    ---

    ## 課程內容

    下列為課程逐字稿／全文（回答**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()

USER_PROMPT_ASK_FAISS_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**回答說明**）牴觸時，**以該兩節為準**。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 作答 user prompt

    {answer_user_prompt_text}

    ## quiz_content（測驗題目）

    {quiz_content}

    ## answer_content（學生先前作答）

    {quiz_answer}

    ## answer_critique（先前批改評語）

    {answer_critique}

    ## 學生追問

    {ask_user_prompt_text}

    ## 回答說明

    - 緊扣**學生追問**，依**課程內容**作答；課程內容未涵蓋者請明說「課程內容未提及」，勿杜撰。
    - 回傳**一段 Markdown 純文字**；**勿**包成 JSON 或鍵值物件。

    ---

    ## 課程內容

    下列為自課程向量庫檢索之片段（回答**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()


def _ask_id_block(exam_quiz_id: int | None, rag_quiz_id: int | None) -> str:
    """關聯識別 Markdown（格式與 answering.py 之 id_block 一致）。"""
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"- **exam_quiz_id**：`{exam_quiz_id}`")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"- **rag_quiz_id**：`{rag_quiz_id}`")
    return ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""


def _ask_llm(api_key: str, user_msg: str, llm_model: str | None) -> str:
    """以 SYSTEM_PROMPT_ASK + user 訊息呼叫 LLM，回傳答案純文字。"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=llm_model or ANSWER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ASK},
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

def run_ask_transcript_only(
    api_key: str,
    transcript: str,
    quiz_content: str,
    ask_user_prompt_text: str,
    *,
    quiz_answer: str = "",
    answer_critique: str = "",
    quiz_user_prompt_text: str = "",
    answer_user_prompt_text: str = "",
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
    llm_model: str | None = None,
) -> str:
    """無 RAG ZIP（unit_type 2／3／4）：system=SYSTEM_PROMPT_ASK；user=USER_PROMPT_ASK_TRANSCRIPT_COURSE。回傳答案純文字。"""
    ts = (transcript or "").strip()
    if not ts:
        raise ValueError("回答用 transcript 未設定")
    user_msg = USER_PROMPT_ASK_TRANSCRIPT_COURSE.format(
        id_block=_ask_id_block(exam_quiz_id, rag_quiz_id),
        quiz_user_prompt_text=_answer_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_answer_field_display(answer_user_prompt_text),
        quiz_content=_answer_field_display(quiz_content),
        quiz_answer=_answer_field_display(quiz_answer),
        answer_critique=_answer_field_display(answer_critique),
        ask_user_prompt_text=_answer_field_display(ask_user_prompt_text),
        context_md=_context_as_markdown_fenced(ts),
    )
    return _ask_llm(api_key, user_msg, llm_model)


# ---------------------------------------------------------------------------
# RAG ZIP 回答（FAISS 或自講義建臨時向量庫）
# ---------------------------------------------------------------------------

def run_ask_job(
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    ask_user_prompt_text: str,
    *,
    quiz_answer: str = "",
    answer_critique: str = "",
    quiz_user_prompt_text: str = "",
    answer_user_prompt_text: str = "",
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
    unit_type: int = 0,
    llm_model: str | None = None,
) -> str:
    """
    在給定的 work_dir（已含 ref.zip）執行向量檢索 + LLM 回答。回傳答案純文字。

    work_dir：由呼叫端建立；內含 ref.zip（RAG 或講義壓縮檔）與子目錄 extract（解壓目標）。
    檢索查詢以「題幹 + 學生追問」串接，較貼近學生問題語意。
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

    embeddings = OpenAIEmbeddings(model=ANSWER_EMBEDDING_MODEL, api_key=api_key)

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
            chunk_size=ANSWER_RAG_CHUNK_SIZE,
            chunk_overlap=ANSWER_RAG_CHUNK_OVERLAP,
        )
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    retrieval_query = "\n\n".join(
        s for s in [(quiz_content or "").strip(), (ask_user_prompt_text or "").strip()] if s
    ) or (quiz_content or ask_user_prompt_text or "")
    retriever = vectorstore.as_retriever(search_kwargs={"k": ANSWER_RETRIEVAL_K})
    docs = retriever.invoke(retrieval_query)
    context_text = "\n\n".join([d.page_content for d in docs])
    context_md = _context_as_markdown_fenced(context_text)

    user_msg = USER_PROMPT_ASK_FAISS_COURSE.format(
        id_block=_ask_id_block(exam_quiz_id, rag_quiz_id),
        quiz_user_prompt_text=_answer_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_answer_field_display(answer_user_prompt_text),
        quiz_content=_answer_field_display(quiz_content),
        quiz_answer=_answer_field_display(quiz_answer),
        answer_critique=_answer_field_display(answer_critique),
        ask_user_prompt_text=_answer_field_display(ask_user_prompt_text),
        context_md=context_md,
    )
    return _ask_llm(api_key, user_msg, llm_model)
