"""
Bank（測試題庫）批改管線 — **bank 專屬，與 rag／exam 完全無關**（prompt 與管線皆獨立）。

- run_bank_answer_job_transcript_only：unit_type 2／3／4，以逐字稿純 LLM 批改。
- run_bank_answer_job：有 FAISS RAG ZIP（或自講義建臨時向量庫）檢索後批改。
- run_bank_answer_job_background：BackgroundTasks 入口；結果存記憶體 results_store，DB 寫入由 insert_answer_fn 注入。

模型 JSON 僅含 answer_critique（評語，無數值分數）；展開為 quiz_comments 後合併純文字寫入 Bank_QA.answer_critique。
"""

import json
import logging
import os
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Callable

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from utils.llm_error import LlmCallError, format_llm_error, is_llm_call_error
from utils.bank_faiss import process_zip_to_docs
from services.bank_generation import BANK_QUIZ_LLM_MODEL, context_as_markdown_fenced

_logger = logging.getLogger("services.bank_answering")

BANK_ANSWER_LLM_MODEL = BANK_QUIZ_LLM_MODEL
BANK_ANSWER_EMBEDDING_MODEL = "text-embedding-3-small"
BANK_ANSWER_RETRIEVAL_K = 5
BANK_ANSWER_RAG_CHUNK_SIZE = 1000
BANK_ANSWER_RAG_CHUNK_OVERLAP = 200


# -----------------------------------------------------------------------------
# LLM 批改 Prompt（bank 專屬，可自由修改）
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_BANK_ANSWER = textwrap.dedent("""
    # 角色

    你是一位教授，請批改學生本題測驗作答。

    ## 指令優先級（必須遵守）

    - 使用者訊息中的 **`## 出題 user prompt`** 與 **`## 作答 user prompt`** 為教師下給你的**直接指令**，優先級**高於**題幹（`quiz_content`）、課程引用與使用者訊息內「批改說明」等泛化規則。
    - 兩節內文有**實質要求**時（非僅空白或占位句如「（未提供）」），**必須完整遵守**，不得忽略、弱化或改寫其意圖。
    - **批改產出**（如何評論、語氣、結構、`quiz_comments`／`text` 之用法）以 **作答 user prompt** 為準；**題意、命題焦點與教學脈絡**並依 **出題 user prompt** 理解與落實。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `answer_critique`：**物件**（頂層**僅**此鍵；**物件內**為 `quiz_comments`：Markdown 字串陣列，**與／或** `text`：Markdown 字串。**勿**出現鍵 `grade`、`quiz_grade`、`score`。）
    """).strip()

USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**批改說明**）牴觸時，**以該兩節為準**。
    - 兩節有實質文字時**務必落實**；**批改寫法**以 **作答 user prompt** 為主，**出題意圖**以 **出題 user prompt** 為主。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 作答 user prompt

    {answer_user_prompt_text}

    ## quiz_content（測驗題目）

    {quiz_content}

    ## quiz_answer（學生作答）

    {quiz_answer}

    ## 批改說明

    - **如何批改、語氣／結構、`quiz_comments` 與 `text` 的用法**請依 **作答 user prompt**（**`## 作答 user prompt`**）；該區無實質文字（含僅「（未提供）」）時再斟酌題幹、**課程內容**與學生作答。
    - **`answer_critique` 物件內**之 `quiz_comments`、`text` 為 **Markdown**。
    - **勿**數值評分；**勿**使用鍵 `grade`、`quiz_grade`、`score`。

    ---

    ## 課程內容

    下列為課程逐字稿／全文（批改**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()

USER_PROMPT_BANK_ANSWER_FAISS_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**批改說明**）牴觸時，**以該兩節為準**。
    - 兩節有實質文字時**務必落實**；**批改寫法**以 **作答 user prompt** 為主，**出題意圖**以 **出題 user prompt** 為主。

    ## 出題 user prompt

    {quiz_user_prompt_text}

    ## 作答 user prompt

    {answer_user_prompt_text}

    ## quiz_content（測驗題目）

    {quiz_content}

    ## quiz_answer（學生作答）

    {quiz_answer}

    ## 批改說明

    - **如何批改、語氣／結構、`quiz_comments` 與 `text` 的用法**請依 **作答 user prompt**（**`## 作答 user prompt`**）；該區無實質文字（含僅「（未提供）」）時再斟酌題幹、**課程內容**與學生作答。
    - **`answer_critique` 物件內**之 `quiz_comments`、`text` 為 **Markdown**。
    - **勿**數值評分；**勿**使用鍵 `grade`、`quiz_grade`、`score`。

    ---

    ## 課程內容

    下列為自課程向量庫檢索之片段（批改**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()


def _answer_field_display(raw: str) -> str:
    return (raw or "").strip() or "（未提供）"


# ---------------------------------------------------------------------------
# LLM JSON 解析輔助（不呼叫 LLM）
# ---------------------------------------------------------------------------

def normalize_answering_llm_json(llm_json: dict[str, Any]) -> None:
    if "quiz_comments" not in llm_json and "comments" in llm_json:
        llm_json["quiz_comments"] = llm_json.pop("comments")


def ingest_llm_answer_response(llm_json: dict[str, Any]) -> None:
    """頂層 answer_critique（物件或字串）展平為 quiz_comments；不保留任何數值評分鍵。"""
    if not isinstance(llm_json, dict):
        return
    raw_comments: Any = None
    ac = llm_json.get("answer_critique")
    if isinstance(ac, dict):
        raw_comments = ac.get("quiz_comments")
        if raw_comments is None and isinstance(ac.get("comments"), list):
            raw_comments = ac["comments"]
        if raw_comments is None and isinstance(ac.get("comments"), str):
            raw_comments = [ac["comments"]]
        if raw_comments is None and isinstance(ac.get("text"), str) and ac.get("text", "").strip():
            raw_comments = [ac["text"].strip()]
    elif isinstance(ac, str):
        s = ac.strip()
        raw_comments = [s] if s else []
    if raw_comments is None:
        raw_comments = llm_json.get("quiz_comments")
    if isinstance(raw_comments, str):
        raw_comments = [raw_comments] if raw_comments.strip() else []
    if not isinstance(raw_comments, list):
        raw_comments = []
    llm_json["quiz_comments"] = raw_comments
    normalize_answering_llm_json(llm_json)
    llm_json.pop("quiz_grade", None)
    llm_json.pop("score", None)


def quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    raw: Any = llm_json.get("quiz_comments")
    if raw is None:
        ac = llm_json.get("answer_critique")
        if isinstance(ac, dict):
            raw = ac.get("quiz_comments")
            if raw is None:
                raw = ac.get("comments")
            if isinstance(raw, str):
                raw = [raw]
        elif isinstance(ac, str):
            s = ac.strip()
            raw = [s] if s else []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            c = x.get("quiz_comment") or x.get("comment") or x.get("criteria")
            if c is not None:
                out.append(str(c))
        elif x is not None:
            out.append(str(x))
    return out


def answer_critique_plain_text_from_result(result_dict: dict[str, Any]) -> str:
    """quiz_comments 合併為單一字串，供寫入 Bank_QA.answer_critique（無 JSON 包殼）。"""
    parts = [c.strip() for c in quiz_comments_from_llm_json(result_dict) if isinstance(c, str) and c.strip()]
    return "\n\n".join(parts)


def cleanup_answer_workspace(work_dir: Path) -> None:
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _bank_qa_id_block(bank_qa_id: int | None) -> str:
    if bank_qa_id is not None and bank_qa_id > 0:
        return f"## 關聯識別\n\n- **bank_qa_id**：`{bank_qa_id}`\n\n"
    return ""


# ---------------------------------------------------------------------------
# 逐字稿純 LLM 批改
# ---------------------------------------------------------------------------

def run_bank_answer_job_transcript_only(
    api_key: str,
    transcript: str,
    quiz_content: str,
    quiz_answer: str,
    *,
    quiz_user_prompt_text: str = "",
    answer_user_prompt_text: str = "",
    bank_qa_id: int | None = None,
    llm_model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    ts = (transcript or "").strip()
    if not ts:
        raise ValueError("批改用 transcript 未設定")
    context_md = context_as_markdown_fenced(ts)
    user_msg = USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE.format(
        id_block=_bank_qa_id_block(bank_qa_id),
        quiz_user_prompt_text=_answer_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_answer_field_display(answer_user_prompt_text),
        quiz_content=_answer_field_display(quiz_content),
        quiz_answer=_answer_field_display(quiz_answer),
        context_md=context_md,
    )
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=llm_model or BANK_ANSWER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_BANK_ANSWER},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as e:
        if is_llm_call_error(e):
            raise LlmCallError(format_llm_error(e)) from e
        raise
    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    ingest_llm_answer_response(llm_json)
    return llm_raw, llm_json


# ---------------------------------------------------------------------------
# RAG ZIP 批改
# ---------------------------------------------------------------------------

def run_bank_answer_job(
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    *,
    answer_user_prompt_text: str = "",
    quiz_user_prompt_text: str = "",
    bank_qa_id: int | None = None,
    unit_type: int = 0,
    llm_model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    zip_source_path = work_dir / "ref.zip"
    extract_folder = work_dir / "extract"
    extract_folder.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(zip_source_path):
        raise ValueError("無效的 ZIP 檔")

    with zipfile.ZipFile(zip_source_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    is_rag_db = False
    db_folder = None
    for root, _, files in os.walk(extract_folder):
        if "index.faiss" in files and "index.pkl" in files:
            is_rag_db = True
            db_folder = root
            break

    embeddings = OpenAIEmbeddings(model=BANK_ANSWER_EMBEDDING_MODEL, api_key=api_key)

    if is_rag_db:
        vectorstore = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)
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
    docs = retriever.invoke(quiz_content)
    context_text = "\n\n".join([d.page_content for d in docs])
    context_md = context_as_markdown_fenced(context_text)

    prompt = USER_PROMPT_BANK_ANSWER_FAISS_COURSE.format(
        id_block=_bank_qa_id_block(bank_qa_id),
        quiz_user_prompt_text=_answer_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_answer_field_display(answer_user_prompt_text),
        quiz_content=_answer_field_display(quiz_content),
        quiz_answer=_answer_field_display(quiz_answer),
        context_md=context_md,
    )

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=llm_model or BANK_ANSWER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_BANK_ANSWER},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as e:
        if is_llm_call_error(e):
            raise LlmCallError(format_llm_error(e)) from e
        raise
    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    ingest_llm_answer_response(llm_json)
    return llm_raw, llm_json


# ---------------------------------------------------------------------------
# 背景批改入口
# ---------------------------------------------------------------------------

def run_bank_answer_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    results_store: dict[str, dict[str, Any]],
    insert_answer_fn: Callable[[dict, str], tuple[str, int] | None],
    *,
    answer_user_prompt_text: str = "",
    bank_qa_id: int | None = None,
    unit_type: int = 0,
    transcript_answer: str | None = None,
    quiz_user_prompt_text: str = "",
    llm_model: str | None = None,
) -> None:
    """執行批改 → 可選寫入 DB → 結果存 results_store。transcript_answer 非空時走逐字稿純 LLM 批改。"""
    try:
        if (transcript_answer or "").strip():
            _, llm_json = run_bank_answer_job_transcript_only(
                api_key,
                transcript_answer.strip(),
                quiz_content,
                quiz_answer,
                quiz_user_prompt_text=quiz_user_prompt_text,
                answer_user_prompt_text=answer_user_prompt_text,
                bank_qa_id=bank_qa_id,
                llm_model=llm_model,
            )
        else:
            _, llm_json = run_bank_answer_job(
                work_dir,
                api_key,
                quiz_content,
                quiz_answer,
                answer_user_prompt_text=answer_user_prompt_text,
                quiz_user_prompt_text=quiz_user_prompt_text,
                bank_qa_id=bank_qa_id,
                unit_type=unit_type,
                llm_model=llm_model,
            )
        result_dict = {"quiz_comments": quiz_comments_from_llm_json(llm_json)}
        inserted = insert_answer_fn(result_dict, quiz_answer)
        if inserted:
            result_dict[inserted[0]] = inserted[1]
            results_store[job_id] = {"status": "ready", "result": result_dict, "error": None, "llm_error": None}
            _logger.info("Bank 批改完成 job_id=%s: %s", job_id, json.dumps(result_dict, ensure_ascii=False))
        else:
            err_detail = (
                "批改 LLM 完成但寫入 DB 失敗。常見原因：未設定 SUPABASE_SERVICE_ROLE_KEY 而遭 RLS 擋、"
                "bank_qa_id 無對應列或已刪除、或欄位 answer_content／answer_critique 與表不符。請見伺服器日誌。"
            )
            _logger.warning("Bank 批改 LLM 已完成但寫入失敗 job_id=%s：%s", job_id, err_detail)
            results_store[job_id] = {"status": "error", "result": None, "error": err_detail, "llm_error": None}
    except LlmCallError as e:
        # LLM 批改失敗屬「批改未完成」，與「LLM 完成但寫入 DB 失敗」一致回 status=error，
        # 避免前端把 status=ready + 空 quiz_comments 誤判為「批改成功但無評語」。
        msg = format_llm_error(e)
        results_store[job_id] = {"status": "error", "result": None, "error": msg, "llm_error": msg}
        _logger.error("Bank 批改 LLM 失敗 job_id=%s: %s", job_id, msg)
    except Exception as e:
        results_store[job_id] = {"status": "error", "result": None, "error": str(e), "llm_error": None}
        _logger.error("Bank 批改失敗 job_id=%s: %s", job_id, e, exc_info=True)
    finally:
        cleanup_answer_workspace(work_dir)
