"""
評分管線 service。包含 LLM JSON 解析、RAG ZIP 批改、逐字稿批改、DB 寫回輔助。
routers/grade.py 與 routers/exam.py 共用，不含任何 FastAPI 路由。

檔案結構（由上而下）：
1. 模型／檢索（及講義臨時向量）常數
2. **LLM 批改 Prompt**（對齊 `quiz_generation`：`SYSTEM_PROMPT_GRADE`、`USER_PROMPT_GRADE_TRANSCRIPTION_COURSE`、`USER_PROMPT_GRADE_FAISS_COURSE`）
3. `_grade_field_display`、LLM JSON 解析輔助、暫存清理、批改與 DB 寫回

重要（維持行為時請留意）：
- 批改 LLM 使用 `response_format=json_object`；模板須含「json」字樣。模型頂層 JSON **僅**含 `answer_critique`（純評語，無分數欄）；管線展開為 **`quiz_comments`** 後，**寫入 DB 之 `answer_critique` 為合併後純文字**（非 `{"quiz_comments":[…]}`）；記憶體 job `result` 仍含 `quiz_comments` 陣列供輪詢（**不依數值評分**）。
- run_grade_job_background：transcription_grade 非空時不走向量庫，與有 FAISS 路徑互斥。
- Rag_Quiz／Exam_Quiz 之 `answer_critique` 皆為評語純文字；不包含數值評分。
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

from postgrest.exceptions import APIError

from utils.datetime_utils import now_taipei_iso
from utils.quiz_generation import _context_as_markdown_fenced
from utils.rag_faiss_zip import process_zip_to_docs
from utils.supabase_client import get_supabase

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 模型與檢索（及講義臨時向量）常數
# -----------------------------------------------------------------------------
# 與出題 `utils.quiz_generation` 共用同一 embedding 模型，維持索引維度一致。
# `GRADE_RETRIEVAL_K`：有 FAISS 或臨時向量時，以題幹為查詢之檢索段數。
# chunk 參數僅於「ZIP 無 FAISS、改由講義建臨時向量庫」時使用。

GRADE_LLM_MODEL = "gpt-4o"
GRADE_EMBEDDING_MODEL = "text-embedding-3-small"
GRADE_RETRIEVAL_K = 5
GRADE_RAG_CHUNK_SIZE = 1000
GRADE_RAG_CHUNK_OVERLAP = 200


# -----------------------------------------------------------------------------
# LLM 批改 Prompt（與「出題」`utils/quiz_generation` 相同分段：`SYSTEM_PROMPT_*` → `USER_PROMPT_*_COURSE`、`---`、`## 課程內容`；user 僅 `.format(...)`）
# -----------------------------------------------------------------------------
# Chat messages：
#   role=system … SYSTEM_PROMPT_GRADE
#   role=user … USER_PROMPT_GRADE_TRANSCRIPTION_COURSE 或 USER_PROMPT_GRADE_FAISS_COURSE；
#   其中 `{context_md}` 僅由逐字稿／向量檢索經 `_context_as_markdown_fenced` 產生，`{id_block}`、`quiz_*` 占位同一 `.format` 填入。
#

SYSTEM_PROMPT_GRADE = textwrap.dedent("""
    # 角色

    你是一位教授，請批改學生本題測驗作答。

    ## 指令優先級（必須遵守）

    - 使用者訊息中的 **`## 出題 user prompt`** 與 **`## 作答 user prompt`** 為教師下給你的**直接指令**，優先級**高於**題幹（`quiz_content`）、課程引用與使用者訊息內「批改說明」等泛化規則。
    - 兩節內文有**實質要求**時（非僅空白或占位句如「（未提供）」），**必須完整遵守**，不得忽略、弱化或改寫其意圖。
    - **批改產出**（如何評論、語氣、結構、`quiz_comments`／`text` 之用法）以 **作答 user prompt** 為準；**題意、命題焦點與教學脈絡**並依 **出題 user prompt** 理解與落实。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `answer_critique`：**物件**（頂層**僅**此鍵；**物件內**為 `quiz_comments`：Markdown 字串陣列，**與／或** `text`：Markdown 字串。**勿**出現鍵 `grade`、`quiz_grade`、`score`。）
    """).strip()

USER_PROMPT_GRADE_TRANSCRIPTION_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**批改說明**）牴觸時，**以該兩節為準**。
    - 兩節有實質文字時**務必落实**；**批改寫法**以 **作答 user prompt** 為主，**出題意圖**以 **出題 user prompt** 為主。

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

USER_PROMPT_GRADE_FAISS_COURSE = textwrap.dedent("""
    {id_block}## 必須遵守（最高優先）

    - 緊接於下的 **`## 出題 user prompt`**、**`## 作答 user prompt`** 兩節內文為本任務**最重要**之依據；與本訊息後段（題幹、學生作答、課程引用、**批改說明**）牴觸時，**以該兩節為準**。
    - 兩節有實質文字時**務必落实**；**批改寫法**以 **作答 user prompt** 為主，**出題意圖**以 **出題 user prompt** 為主。

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


def _grade_field_display(raw: str) -> str:
    """填入批改 user 模板之欄位；空則與舊版一致顯示「（未提供）」。"""
    return (raw or "").strip() or "（未提供）"


# ---------------------------------------------------------------------------
# LLM JSON 解析輔助
# ---------------------------------------------------------------------------
# 以下函式不呼叫 LLM，僅處理「模型已回傳」或「DB 已儲存」的 JSON／欄位，供路由與寫回共用。

def clamp_quiz_grade(v: Any) -> int:
    """將任意型別之分數化為 0～5 整數；無法解析則 0（與 DB／API 滿分一致）。"""
    if v is None:
        return 0
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, n))


def quiz_grade_from_llm_json(llm_json: dict[str, Any]) -> int:
    """批改不再產出數值分數；永遠回傳 0（相容舊呼叫介面）。"""
    _ = llm_json
    return 0


def normalize_grading_llm_json(llm_json: dict[str, Any]) -> None:
    """就地修改：舊鍵 comments → quiz_comments（與前端／DB 欄位命名一致）。"""
    if "quiz_comments" not in llm_json and "comments" in llm_json:
        llm_json["quiz_comments"] = llm_json.pop("comments")


def ingest_llm_grade_response(llm_json: dict[str, Any]) -> None:
    """
    將頂層 `answer_critique`（評語物件或字串）展平為 `quiz_comments`；**不依賴、不保留**任何數值評分鍵。
    仍相容頂層僅含 `quiz_comments` 之舊回傳（忽略其中之 quiz_grade／score）。
    """
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
    normalize_grading_llm_json(llm_json)

    llm_json.pop("quiz_grade", None)
    llm_json.pop("score", None)


def quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    """
    自 LLM JSON 取出 quiz_comments，正規化為字串列表。

    相容尚未展平：僅有 answer_critique 物件時由其內讀取 quiz_comments／comments／text。

    元素可為 str，或 dict（取 quiz_comment／comment／criteria）；其餘型別轉 str。
    """
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
            # 相容模型以物件陣列回傳多則評語的寫法
            c = x.get("quiz_comment") or x.get("comment") or x.get("criteria")
            if c is not None:
                out.append(str(c))
        elif x is not None:
            out.append(str(x))
    return out


def answer_critique_plain_text_from_result(result_dict: dict[str, Any]) -> str:
    """將 `quiz_comments` 合併為單一字串，供寫入 `answer_critique`（無 JSON 包殼）。"""
    parts = [c.strip() for c in quiz_comments_from_llm_json(result_dict) if isinstance(c, str) and c.strip()]
    return "\n\n".join(parts)


def quiz_grade_from_answer_critique(critique_raw: Any) -> int | None:
    """
    自 answer_critique（JSON 字串或 dict）解析 quiz_grade；失敗則 None。

    讀取順序：頂層 quiz_grade → quiz_grade_metadata 內 quiz_grade／score。
    用於 update_* 寫入後「讀回驗證」與前端顯示分數一致性檢查。
    """
    if critique_raw is None:
        return None
    try:
        data: Any
        if isinstance(critique_raw, dict):
            data = critique_raw
        else:
            s = str(critique_raw).strip()
            if not s:
                return None
            data = json.loads(s)
        if not isinstance(data, dict):
            return None
        g = data.get("quiz_grade")
        if g is None:
            meta = data.get("quiz_grade_metadata")
            if isinstance(meta, dict):
                g = meta.get("quiz_grade", meta.get("score"))
        if g is None:
            return None
        return int(round(float(g)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def critique_stored_grade_matches(critique_raw: Any, expected: int) -> bool:
    """寫回後一致性檢查；目前不依數字評分，預期分數為 0 時通過。"""
    if critique_raw is None:
        return False
    if int(expected) == 0:
        return True
    g = quiz_grade_from_answer_critique(critique_raw)
    if g is None:
        return True
    return int(g) == int(expected)


# ---------------------------------------------------------------------------
# 暫存目錄清理
# ---------------------------------------------------------------------------

def cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄（含 ref.zip 解壓內容）；ignore_errors 避免因權限略過失敗。"""
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 逐字稿純 LLM 批改（unit_type 2／3／4，不讀 RAG ZIP）
# ---------------------------------------------------------------------------

def run_grade_job_transcription_only(
    api_key: str,
    transcription: str,
    quiz_content: str,
    quiz_answer: str,
    *,
    quiz_user_prompt_text: str = "",
    answer_user_prompt_text: str = "",
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    無 RAG ZIP（unit_type 2／3／4）：system = SYSTEM_PROMPT_GRADE；user = USER_PROMPT_GRADE_TRANSCRIPTION_COURSE，
    其中課程逐字稿／全文經 `_context_as_markdown_fenced` 填入 `{context_md}`。
    回傳 (LLM 訊息原文, 解析後 JSON)。

    quiz_user_prompt_text／answer_user_prompt_text：與 **USER_PROMPT_GRADE_TRANSCRIPTION_COURSE** 占位符同名，空則經 `_grade_field_display` 為「（未提供）」。
    """
    ts = (transcription or "").strip()
    if not ts:
        raise ValueError("批改用 transcription 未設定")

    # 關聯識別 Markdown（格式須與 run_grade_job 一致；供 USER_PROMPT_GRADE_TRANSCRIPTION_COURSE）
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"- **exam_quiz_id**：`{exam_quiz_id}`")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"- **rag_quiz_id**：`{rag_quiz_id}`")
    id_block = ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""

    context_md = _context_as_markdown_fenced(ts)
    # system／user 與 utils.quiz_generation 出題路徑一致：規範在 system；題目／欄位／課程內文在 user。
    user_msg = USER_PROMPT_GRADE_TRANSCRIPTION_COURSE.format(
        id_block=id_block,
        quiz_user_prompt_text=_grade_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_grade_field_display(answer_user_prompt_text),
        quiz_content=_grade_field_display(quiz_content),
        quiz_answer=_grade_field_display(quiz_answer),
        context_md=context_md,
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GRADE_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GRADE},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        # 理論上 json_object 不應回非 JSON；防禦性處理避免背景 job 整段崩潰。
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    ingest_llm_grade_response(llm_json)
    return llm_raw, llm_json


# ---------------------------------------------------------------------------
# RAG ZIP 批改
# ---------------------------------------------------------------------------

def run_grade_job(
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    answer_user_prompt_text: str = "",
    *,
    quiz_user_prompt_text: str = "",
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
    unit_type: int = 0,
) -> tuple[str, dict[str, Any]]:
    """
    在給定的 work_dir（已含 ref.zip）執行向量檢索 + GPT 評分。回傳 (LLM 訊息原文, 解析後 JSON)。

    work_dir：由路由建立；內含 ref.zip（RAG 或講義壓縮檔）與子目錄 extract（解壓目標）。
    unit_type：僅在「無 FAISS、改由講義建臨時向量庫」時傳入 process_zip_to_docs。
    quiz_user_prompt_text：與 **USER_PROMPT_GRADE_FAISS_COURSE** 占位符同名；空則經 `_grade_field_display` 顯示「（未提供）」。
    LLM 訊息：`SYSTEM_PROMPT_GRADE`（system）+ `USER_PROMPT_GRADE_FAISS_COURSE`（user，其中 `{context_md}` 為 `_context_as_markdown_fenced(檢索片段)`）。
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

    embeddings = OpenAIEmbeddings(model=GRADE_EMBEDDING_MODEL, api_key=api_key)

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
            chunk_size=GRADE_RAG_CHUNK_SIZE,
            chunk_overlap=GRADE_RAG_CHUNK_OVERLAP,
        )
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 以題幹當檢索查詢（與 utils.quiz_generation 固定查詢句不同，較貼近本題語意）。
    retriever = vectorstore.as_retriever(search_kwargs={"k": GRADE_RETRIEVAL_K})
    docs = retriever.invoke(quiz_content)
    context_text = "\n\n".join([d.page_content for d in docs])
    context_md = _context_as_markdown_fenced(context_text)

    # 關聯識別 Markdown（格式須與 run_grade_job_transcription_only 一致；供 USER_PROMPT_GRADE_FAISS_COURSE）
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"- **exam_quiz_id**：`{exam_quiz_id}`")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"- **rag_quiz_id**：`{rag_quiz_id}`")
    id_block = ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""

    prompt = USER_PROMPT_GRADE_FAISS_COURSE.format(
        id_block=id_block,
        quiz_user_prompt_text=_grade_field_display(quiz_user_prompt_text),
        answer_user_prompt_text=_grade_field_display(answer_user_prompt_text),
        quiz_content=_grade_field_display(quiz_content),
        quiz_answer=_grade_field_display(quiz_answer),
        context_md=context_md,
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GRADE_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GRADE},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        # 與逐字稿路徑相同防禦；json_object 仍應保證可 parse。
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    ingest_llm_grade_response(llm_json)
    return llm_raw, llm_json


# ---------------------------------------------------------------------------
# 通用背景評分入口
# ---------------------------------------------------------------------------
# 由路由註冊 BackgroundTasks 呼叫；不直接對外暴露 HTTP。
# results_store：記憶體 dict，鍵為 job_id；供 GET .../grade-result 輪詢。

def run_grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    results_store: dict[str, dict[str, Any]],
    insert_answer_fn: Callable[[dict, str], tuple[str, int] | None],
    answer_user_prompt_text: str = "",
    *,
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
    unit_type: int = 0,
    transcription_grade: str | None = None,
    quiz_user_prompt_text: str = "",
) -> None:
    """
    通用背景評分：執行評分、可選寫入 DB、結果存 results_store。
    insert_answer_fn(result_dict, quiz_answer) 寫入 DB 並回傳 (id_key, id_val) 或 None。
    transcription_grade 非空時改走逐字稿純 LLM 批改（不讀 RAG ZIP）。
    """
    try:
        # 與 unit_type 2/3/4 批改一致：有逐字稿字串則不開 ref.zip 向量流程（避免重複讀 ZIP）。
        if (transcription_grade or "").strip():
            _, llm_json = run_grade_job_transcription_only(
                api_key,
                transcription_grade.strip(),
                quiz_content,
                quiz_answer,
                quiz_user_prompt_text=quiz_user_prompt_text,
                answer_user_prompt_text=answer_user_prompt_text,
                exam_quiz_id=exam_quiz_id,
                rag_quiz_id=rag_quiz_id,
            )
        else:
            _, llm_json = run_grade_job(
                work_dir,
                api_key,
                quiz_content,
                quiz_answer,
                answer_user_prompt_text,
                quiz_user_prompt_text=quiz_user_prompt_text,
                exam_quiz_id=exam_quiz_id,
                rag_quiz_id=rag_quiz_id,
                unit_type=unit_type,
            )
        # 與 API 回傳欄位對齊；insert_answer_fn 內可能再寫入 critique／分數至 DB。
        result_dict = {
            "quiz_comments": quiz_comments_from_llm_json(llm_json),
        }
        inserted = insert_answer_fn(result_dict, quiz_answer)
        if inserted:
            result_dict[inserted[0]] = inserted[1]
            if inserted[0] == "rag_quiz_id":
                result_dict["rag_answer_id"] = inserted[1]
            results_store[job_id] = {"status": "ready", "result": result_dict, "error": None}
            _logger.info(
                "批改完成 job_id=%s 回傳結果: %s",
                job_id,
                json.dumps(result_dict, ensure_ascii=False),
            )
        else:
            err_detail = (
                "更新 Rag_Quiz 評分欄位失敗。常見原因：未設定 SUPABASE_SERVICE_ROLE_KEY 而改用 anon 遭 RLS 擋、"
                "rag_quiz_id 無對應列或已刪除、或欄位 quiz_content／answer_user_prompt_text／answer_content／answer_critique 與表不符。請見伺服器日誌。"
            )
            _logger.warning("批改 LLM 已完成但寫入答案表失敗 job_id=%s：%s", job_id, err_detail)
            results_store[job_id] = {"status": "error", "result": None, "error": err_detail}
    except Exception as e:
        results_store[job_id] = {"status": "error", "result": None, "error": str(e)}
        _logger.error("批改失敗 job_id=%s: %s", job_id, e, exc_info=True)
    finally:
        cleanup_grade_workspace(work_dir)


# ---------------------------------------------------------------------------
# DB 寫回輔助
# ---------------------------------------------------------------------------
# 皆使用 get_supabase()；若環境僅 anon key，RLS 可能導致 update 成功但 select 無列，需看 log。

def _rag_quiz_missing_column_error(exc: BaseException, column: str) -> bool:
    """PostgREST PGRST204：請求 body／欄位清單含資料表不存在的欄位。"""
    col = column.strip()
    if isinstance(exc, APIError):
        msg = exc.message or ""
        return (exc.code or "") == "PGRST204" and col in msg
    text = str(exc)
    return "PGRST204" in text and col in text


def update_rag_quiz_with_grade(
    result_dict: dict,
    quiz_answer: str,
    *,
    rag_quiz_id: int,
    answer_user_prompt_text: str = "",
    quiz_content: str = "",
) -> tuple[str, int] | None:
    """更新 public.Rag_Quiz；`answer_critique` 存評語純文字（與 Exam_Quiz 一致）；成功後讀回驗證（舊表無該欄時走降級路徑）。"""
    if rag_quiz_id <= 0:
        return None
    ts = now_taipei_iso()
    qc_persist = (quiz_content or "").strip()
    row: dict[str, Any] = {
        "answer_user_prompt_text": (answer_user_prompt_text or "").strip(),
        "answer_content": quiz_answer or "",
        "answer_critique": answer_critique_plain_text_from_result(result_dict),
        "updated_at": ts,
    }
    # 僅在呼叫端傳入非空 quiz_content 時一併更新題幹（避免空字串蓋掉既有題目）。
    if qc_persist:
        row = {"quiz_content": qc_persist, **row}
    try:
        supabase = get_supabase()
        try:
            supabase.table("Rag_Quiz").update(row).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        except Exception as first_err:
            # 部分環境尚無 answer_critique 欄位（PGRST204）：略過該欄仍回傳 rag_quiz_id。
            if not _rag_quiz_missing_column_error(first_err, "answer_critique"):
                raise
            row_lean = {k: v for k, v in row.items() if k != "answer_critique"}
            supabase.table("Rag_Quiz").update(row_lean).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
            _logger.warning(
                "Rag_Quiz 無 answer_critique 欄位，已略過評語寫入（rag_quiz_id=%s）",
                rag_quiz_id,
            )
            chk = (
                supabase.table("Rag_Quiz")
                .select("quiz_content")
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
            if not chk.data:
                _logger.warning(
                    "Rag_Quiz update 後讀不到列（rag_quiz_id=%s 不存在、已刪除或遭 RLS 擋）",
                    rag_quiz_id,
                )
                return None
            cr0 = chk.data[0]
            if qc_persist and (cr0.get("quiz_content") or "").strip() != qc_persist:
                _logger.warning(
                    "Rag_Quiz 讀回 quiz_content 與預期不符（rag_quiz_id=%s）",
                    rag_quiz_id,
                )
                return None
            return ("rag_quiz_id", rag_quiz_id)
        # 正常路徑：寫入後讀回 answer_critique，確認可解析（無分數欄時僅確認寫入成功）。
        chk = (
            supabase.table("Rag_Quiz")
            .select("answer_critique, quiz_content")
            .eq("rag_quiz_id", rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not chk.data:
            _logger.warning(
                "Rag_Quiz update 後讀不到列（rag_quiz_id=%s 不存在、已刪除或遭 RLS 擋）",
                rag_quiz_id,
            )
            return None
        cr0 = chk.data[0]
        if not critique_stored_grade_matches(cr0.get("answer_critique"), 0):
            _logger.warning(
                "Rag_Quiz 讀回 answer_critique 內數值評分（若有）與預期 0／無分數不符（rag_quiz_id=%s）",
                rag_quiz_id,
            )
            return None
        if qc_persist and (cr0.get("quiz_content") or "").strip() != qc_persist:
            _logger.warning(
                "Rag_Quiz 讀回 quiz_content 與預期不符（rag_quiz_id=%s）",
                rag_quiz_id,
            )
            return None
        return ("rag_quiz_id", rag_quiz_id)
    except Exception as e:
        _logger.warning("Rag_Quiz update 失敗: %s", e, exc_info=True)
    return None


def update_exam_quiz_with_grade(
    result_dict: dict,
    quiz_answer: str,
    *,
    exam_quiz_id: int,
) -> tuple[str, int] | None:
    """更新 public.Exam_Quiz；answer_critique 存評語純文字（`quiz_comments` 合併，非 JSON），成功後讀回驗證。"""
    if exam_quiz_id <= 0:
        return None
    critique = answer_critique_plain_text_from_result(result_dict)
    ts = now_taipei_iso()
    try:
        supabase = get_supabase()
        supabase.table("Exam_Quiz").update({
            "answer_content": quiz_answer or "",
            "answer_critique": critique,
            "updated_at": ts,
        }).eq("exam_quiz_id", exam_quiz_id).execute()
        chk = (
            supabase.table("Exam_Quiz")
            .select("answer_critique")
            .eq("exam_quiz_id", exam_quiz_id)
            .limit(1)
            .execute()
        )
        if not chk.data:
            _logger.warning(
                "Exam_Quiz update 後讀不到列（exam_quiz_id=%s 不存在或遭 RLS 擋）",
                exam_quiz_id,
            )
            return None
        if not critique_stored_grade_matches(chk.data[0].get("answer_critique"), 0):
            _logger.warning(
                "Exam_Quiz 讀回 answer_critique 與無分數預期不符（exam_quiz_id=%s）",
                exam_quiz_id,
            )
            return None
        return ("exam_quiz_id", exam_quiz_id)
    except Exception as e:
        _logger.warning("Exam_Quiz grade update 失敗: %s", e, exc_info=True)
    return None
