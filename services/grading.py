"""
評分管線 service。包含 LLM JSON 解析、RAG ZIP 批改、逐字稿批改、DB 寫回輔助。
routers/grade.py 與 routers/exam.py 共用，不含任何 FastAPI 路由。

檔案結構（建議閱讀順序）：
1. LLM JSON 解析輔助（分數、評語、舊鍵相容、answer_critique 讀回驗證）
2. 暫存目錄清理
3. LLM Prompt 範本（長文集中；與 utils/quiz_generation 出題 prompt 分檔）
4. run_grade_job_transcription_only／run_grade_job（兩條批改路徑）
5. run_grade_job_background（非同步入口）
6. DB 寫回（Rag_Quiz／Exam_Quiz）

重要（維持行為時請留意）：
- 批改 LLM 使用 response_format=json_object；prompt 須含「json」字樣（見 GRADE_RUBRIC／PROMPT_GRADE_JSON_TAIL）。
- run_grade_job_background：transcription_grade 非空時不走向量庫，與有 FAISS 路徑互斥。
- Rag_Quiz 的 answer_critique 內可存整包 LLM 結果（quiz_grade_metadata）；Exam_Quiz 則存精簡 JSON。
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
from utils.rag_faiss_zip import process_zip_to_docs
from utils.supabase_client import get_supabase

_logger = logging.getLogger(__name__)


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
    """自 LLM JSON 取出分數；優先 quiz_grade，若無則相容舊鍵 score（歷史 prompt／模型輸出）。"""
    v = llm_json.get("quiz_grade")
    if v is None:
        v = llm_json.get("score")
    return clamp_quiz_grade(v)


def normalize_grading_llm_json(llm_json: dict[str, Any]) -> None:
    """就地修改：舊鍵 comments → quiz_comments（與前端／DB 欄位命名一致）。"""
    if "quiz_comments" not in llm_json and "comments" in llm_json:
        llm_json["quiz_comments"] = llm_json.pop("comments")


def quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    """
    自 LLM JSON 取出 quiz_comments，正規化為字串列表。

    元素可為 str，或 dict（取 quiz_comment／comment／criteria）；其餘型別轉 str。
    """
    raw = llm_json.get("quiz_comments")
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
    """answer_critique 內解析出之分數是否與預期整數一致（寫回後防呆）。"""
    g = quiz_grade_from_answer_critique(critique_raw)
    return g is not None and g == int(expected)


# ---------------------------------------------------------------------------
# 暫存目錄清理
# ---------------------------------------------------------------------------

def cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄（含 ref.zip 解壓內容）；ignore_errors 避免因權限略過失敗。"""
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# LLM Prompt 範本（集中維護；語氣與 JSON 鍵名請只改本區）
# ---------------------------------------------------------------------------
# PROMPT_GRADE_USER_TRANSCRIPT：佔位符 id_block, qup_disp, aup_disp, qc_disp, qa_disp, rubric（=GRADE_RUBRIC）。
# PROMPT_GRADE_USER_RAG：同上另加 context_text（檢索合併片段）。
# PROMPT_GRADE_JSON_TAIL：逐字稿路徑 system 常無「json」時接在 user 尾端。
# id_block 由呼叫端組 Markdown「## 關聯識別」；空字串時標題列仍正確銜接「## 評分依據」。

GRADE_LLM_MODEL = "gpt-4o"

GRADE_RUBRIC = textwrap.dedent("""
    ## 重要限制

    - 請使用 **繁體中文（Traditional Chinese）** 撰寫 `quiz_comments` 中各則評語。

    ## 評分標準

    `quiz_grade` 為 **0～5 的整數**（滿分 5）：

    - **0**：完全錯誤或未作答。
    - **1**：只有少量內容正確。
    - **2**：大幅缺漏，只有部分內容正確。
    - **3**：部分正確，但有大幅缺漏。
    - **4**：大致正確，略有不足。
    - **5**：完全正確且完整。

    ## 輸出格式（JSON）

    請以 **JSON 物件**回傳，鍵名固定為：

    - `quiz_grade`：整數 0～5
    - `quiz_comments`：字串陣列
    """).strip()

# {id_block}：可為空；有 exam_quiz_id／rag_quiz_id 時為「## 關聯識別」Markdown 區塊（含尾端空行）
PROMPT_GRADE_USER_TRANSCRIPT = textwrap.dedent("""
    # 批改任務

    你是一位教授，請依下列資料批改這道題目。

    {id_block}## 評分依據

    請依下列欄位評分（**不依賴**課程向量庫檢索）：

    1. `quiz_user_prompt_text`（出題補充）
    2. `answer_user_prompt_text`（作答補充／批改指引）
    3. `quiz_content`（測驗題目）
    4. `quiz_answer`（學生作答）

    ## quiz_user_prompt_text（出題補充）

    {qup_disp}

    ## answer_user_prompt_text（作答補充／批改指引）

    {aup_disp}

    ## quiz_content（測驗題目）

    {qc_disp}

    ## quiz_answer（學生作答）

    {qa_disp}

    {rubric}
    """).strip()

# 逐字稿 system 常無「json」字樣時接於 user 末尾
PROMPT_GRADE_JSON_TAIL = textwrap.dedent("""
    ## 輸出格式（JSON）

    請以 **JSON 物件**輸出 `quiz_grade`（整數）與 `quiz_comments`（字串陣列）。
    """).strip()

# 向量檢索批改：單一 user 訊息；{context_text} 為 RAG 檢索合併片段
PROMPT_GRADE_USER_RAG = textwrap.dedent("""
    # 批改任務

    你是一位教授，請依下列資料批改這道題目。

    {id_block}## 評分依據

    請綜合下列 **API 傳入**之測驗題目、學生作答、作答補充／批改指引，以及 **課程內容（RAG 檢索）**，評估學生答案是否正確。

    ## quiz_content（測驗題目）

    {qc_disp}

    ## quiz_answer（學生作答）

    {qa_disp}

    ## answer_user_prompt_text（作答補充／批改指引）

    {aup_disp}

    ## 課程內容（RAG 檢索）

    {context_text}

    {rubric}
    """).strip()


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
    無 RAG ZIP（unit_type 2／3／4）：system = transcription；user 含評分規範。
    回傳 (LLM 訊息原文, 解析後 JSON)。

    quiz_user_prompt_text／answer_user_prompt_text：與路由傳入一致，空則顯示「（未提供）」
    仍保留區塊標題，避免模型誤以為可省略該欄位。
    """
    ts = (transcription or "").strip()
    if not ts:
        raise ValueError("批改用 transcription 未設定")

    qc_disp = (quiz_content or "").strip() or "（未提供）"
    qa_disp = (quiz_answer or "").strip() or "（未提供）"
    qup_disp = (quiz_user_prompt_text or "").strip() or "（未提供）"
    aup_disp = (answer_user_prompt_text or "").strip() or "（未提供）"
    # 關聯識別 Markdown（格式須與 run_grade_job 一致，供 PROMPT_GRADE_USER_* 使用）
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"- **exam_quiz_id**：`{exam_quiz_id}`")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"- **rag_quiz_id**：`{rag_quiz_id}`")
    id_block = ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""

    # system 為課程逐字稿全文；與 run_grade_job（向量檢索）分流，勿混用。
    user_msg = PROMPT_GRADE_USER_TRANSCRIPT.format(
        id_block=id_block,
        qup_disp=qup_disp,
        aup_disp=aup_disp,
        qc_disp=qc_disp,
        qa_disp=qa_disp,
        rubric=GRADE_RUBRIC,
    )
    # OpenAI json_object：整份 messages 須含「json」（逐字稿 system 可能無該字樣）。
    if "json" not in (ts + user_msg).lower():
        user_msg += "\n\n" + PROMPT_GRADE_JSON_TAIL

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GRADE_LLM_MODEL,
        # 逐字稿路徑：課程內容在 system，規範與題目在 user。
        messages=[
            {"role": "system", "content": ts},
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
    normalize_grading_llm_json(llm_json)
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
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
    unit_type: int = 0,
) -> tuple[str, dict[str, Any]]:
    """
    在給定的 work_dir（已含 ref.zip）執行向量檢索 + GPT 評分。回傳 (LLM 訊息原文, 解析後 JSON)。

    work_dir：由路由建立；內含 ref.zip（RAG 或講義壓縮檔）與子目錄 extract（解壓目標）。
    unit_type：僅在「無 FAISS、改由講義建臨時向量庫」時傳入 process_zip_to_docs。
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

    # 與出題模組相同 embedding 模型；維持索引維度一致（勿與 quiz_generation 各走不同模型）。
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

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
        # chunk 參數與向量品質／token 成本權衡；異動時建議連動測試長講義批改。
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 以題幹當檢索查詢（與 utils.quiz_generation 固定查詢句不同，較貼近本題語意）。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(quiz_content)
    context_text = "\n\n".join([d.page_content for d in docs])

    qc_disp = (quiz_content or "").strip() or "（未提供）"
    qa_disp = (quiz_answer or "").strip() or "（未提供）"
    aup_disp = (answer_user_prompt_text or "").strip() or "（未提供）"
    # 關聯識別 Markdown（格式須與 run_grade_job_transcription_only 一致，供 PROMPT_GRADE_USER_* 使用）
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"- **exam_quiz_id**：`{exam_quiz_id}`")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"- **rag_quiz_id**：`{rag_quiz_id}`")
    id_block = ("## 關聯識別\n\n" + "\n".join(id_lines) + "\n\n") if id_lines else ""

    prompt = PROMPT_GRADE_USER_RAG.format(
        id_block=id_block,
        qc_disp=qc_disp,
        qa_disp=qa_disp,
        aup_disp=aup_disp,
        context_text=context_text,
        rubric=GRADE_RUBRIC,
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=GRADE_LLM_MODEL,
        # 單一 user：題幹、作答、檢索片段與 rubric 皆在同一訊息（與逐字稿批改雙訊息不同）。
        messages=[{"role": "user", "content": prompt}],
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
    normalize_grading_llm_json(llm_json)
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
                exam_quiz_id=exam_quiz_id,
                rag_quiz_id=rag_quiz_id,
                unit_type=unit_type,
            )
        # 與 API 回傳欄位對齊；insert_answer_fn 內可能再寫入 critique／分數至 DB。
        result_dict: dict[str, Any] = {
            "quiz_grade": quiz_grade_from_llm_json(llm_json),
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


def answer_row_payload(
    result_dict: dict,
    quiz_answer: str,
    *,
    answer_text_column: str = "answer_content",
) -> dict[str, Any]:
    """
    寫入 Rag_Quiz.answer_critique 的 JSON 本體。

    quiz_grade_metadata：保留完整 LLM dict，供前端還原評語結構；answer_text_column 對齊表欄位名。
    """
    grade = clamp_quiz_grade(result_dict.get("quiz_grade", result_dict.get("score", 0)))
    return {
        answer_text_column: quiz_answer or "",
        "quiz_grade": grade,
        "quiz_grade_metadata": result_dict,
    }


def update_rag_quiz_with_grade(
    result_dict: dict,
    quiz_answer: str,
    *,
    rag_quiz_id: int,
    answer_user_prompt_text: str = "",
    quiz_content: str = "",
) -> tuple[str, int] | None:
    """更新 public.Rag_Quiz；成功後讀回驗證分數／題幹（舊表無 answer_critique 時走降級路徑）。"""
    if rag_quiz_id <= 0:
        return None
    grade = clamp_quiz_grade(result_dict.get("quiz_grade", result_dict.get("score", 0)))
    ts = now_taipei_iso()
    qc_persist = (quiz_content or "").strip()
    row: dict[str, Any] = {
        "answer_user_prompt_text": (answer_user_prompt_text or "").strip(),
        "answer_content": quiz_answer or "",
        "answer_critique": json.dumps(
            answer_row_payload(result_dict, quiz_answer, answer_text_column="answer_content"),
            ensure_ascii=False,
        ),
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
            # 部分環境尚無 answer_critique 欄位（PGRST204）：略過 JSON 欄仍回傳 rag_quiz_id。
            if not _rag_quiz_missing_column_error(first_err, "answer_critique"):
                raise
            row_lean = {k: v for k, v in row.items() if k != "answer_critique"}
            supabase.table("Rag_Quiz").update(row_lean).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
            _logger.warning(
                "Rag_Quiz 無 answer_critique 欄位，已略過評分 JSON 寫入（rag_quiz_id=%s）",
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
        # 正常路徑：寫入後讀回 answer_critique，確認 quiz_grade 與本次 LLM 一致。
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
        if not critique_stored_grade_matches(cr0.get("answer_critique"), grade):
            _logger.warning(
                "Rag_Quiz 讀回 answer_critique 內 quiz_grade 與預期 %s 不符（rag_quiz_id=%s）",
                grade,
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
    """更新 public.Exam_Quiz；answer_critique 存精簡 JSON（分數 + quiz_comments），成功後讀回驗證。"""
    if exam_quiz_id <= 0:
        return None
    grade = clamp_quiz_grade(result_dict.get("quiz_grade", result_dict.get("score", 0)))
    comments = quiz_comments_from_llm_json(result_dict)
    # Exam 路徑 critique 較精簡（無整包 metadata），與 Rag_Quiz 欄位語意分流。
    critique = json.dumps({"quiz_grade": grade, "quiz_comments": comments}, ensure_ascii=False)
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
        if not critique_stored_grade_matches(chk.data[0].get("answer_critique"), grade):
            _logger.warning(
                "Exam_Quiz 讀回 answer_critique 內 quiz_grade 與預期 %s 不符（exam_quiz_id=%s）",
                grade,
                exam_quiz_id,
            )
            return None
        return ("exam_quiz_id", exam_quiz_id)
    except Exception as e:
        _logger.warning("Exam_Quiz grade update 失敗: %s", e, exc_info=True)
    return None
