"""
評分管線 service。包含 LLM JSON 解析、RAG ZIP 批改、逐字稿批改、DB 寫回輔助。
routers/grade.py 與 routers/exam.py 共用，不含任何 FastAPI 路由。
"""

import json
import logging
import os
import shutil
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

def clamp_quiz_grade(v: Any) -> int:
    """將 quiz_grade 化為 0～5 的整數（滿分固定為 5）。"""
    if v is None:
        return 0
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, n))


def quiz_grade_from_llm_json(llm_json: dict[str, Any]) -> int:
    """自 LLM JSON 取出分數；優先 quiz_grade，若無則相容舊鍵 score。"""
    v = llm_json.get("quiz_grade")
    if v is None:
        v = llm_json.get("score")
    return clamp_quiz_grade(v)


def normalize_grading_llm_json(llm_json: dict[str, Any]) -> None:
    """將舊鍵 comments 合併至 quiz_comments 後移除（與 API 欄位名一致）。"""
    if "quiz_comments" not in llm_json and "comments" in llm_json:
        llm_json["quiz_comments"] = llm_json.pop("comments")


def quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    """自 LLM JSON 取出 quiz_comments，正規化為字串列表。"""
    raw = llm_json.get("quiz_comments")
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


def quiz_grade_from_answer_critique(critique_raw: Any) -> int | None:
    """自 answer_critique（JSON 字串或 dict）解析 quiz_grade；失敗則 None（相容 quiz_grade_metadata）。"""
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
    """answer_critique 內 quiz_grade 是否與預期一致。"""
    g = quiz_grade_from_answer_critique(critique_raw)
    return g is not None and g == int(expected)


# ---------------------------------------------------------------------------
# 暫存目錄清理
# ---------------------------------------------------------------------------

def cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄。"""
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 評分 prompt 常數
# ---------------------------------------------------------------------------

_GRADE_RUBRIC = """\
        【重要限制】
        請使用繁體中文 (Traditional Chinese) 撰寫評語 (quiz_comments)。
        【評分標準】
        0-5分，一定是整數 (quiz_grade)。
        0: 完全錯誤或未作答。
        1: 只有少量內容正確。
        2: 大幅缺漏，只有部分內容正確。
        3: 部分正確，但有大幅缺漏。
        4: 大致正確，略有不足。
        5: 完全正確且完整。
        【輸出 JSON】
        請以 JSON 格式回傳：
        { "quiz_grade": int,
        "quiz_comments": str[] }"""


def _id_block(exam_quiz_id: int | None, rag_quiz_id: int | None) -> str:
    lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        lines.append(f"        【exam_quiz_id】{exam_quiz_id}")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        lines.append(f"        【rag_quiz_id】{rag_quiz_id}")
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


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
    """
    ts = (transcription or "").strip()
    if not ts:
        raise ValueError("批改用 transcription 未設定")

    qc_disp = (quiz_content or "").strip() or "（未提供）"
    qa_disp = (quiz_answer or "").strip() or "（未提供）"
    qup_disp = (quiz_user_prompt_text or "").strip() or "（未提供）"
    aup_disp = (answer_user_prompt_text or "").strip() or "（未提供）"
    blk = _id_block(exam_quiz_id, rag_quiz_id)

    user_msg = f"""
        你是一位教授，請批改這道題目。
        {blk}        【評分規範】
        請依下列「出題補充」（quiz_user_prompt_text）、「作答補充／批改指引」（answer_user_prompt_text）、測驗題目（quiz_content）與學生作答（quiz_answer）評分（不依賴課程向量庫檢索）。
        【quiz_user_prompt_text 出題補充】
        {qup_disp}
        【answer_user_prompt_text 作答補充／批改指引】
        {aup_disp}
        【quiz_content 測驗題目】
        {qc_disp}
        【quiz_answer 學生作答】
        {qa_disp}
{_GRADE_RUBRIC}
    """
    if "json" not in (ts + user_msg).lower():
        user_msg += "\n\n請以 JSON 物件輸出 quiz_grade（整數）與 quiz_comments（字串陣列）。"

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
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
    """在給定的 work_dir（已含 ref.zip）執行 RAG + GPT 評分。回傳 (LLM 訊息原文, 解析後 JSON)。"""
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(quiz_content)
    context_text = "\n\n".join([d.page_content for d in docs])

    qc_disp = (quiz_content or "").strip() or "（未提供）"
    qa_disp = (quiz_answer or "").strip() or "（未提供）"
    aup_disp = (answer_user_prompt_text or "").strip() or "（未提供）"
    blk = _id_block(exam_quiz_id, rag_quiz_id)

    prompt = f"""
        你是一位教授，請批改這道題目。
        {blk}        【評分規範】
        請依下列「API 傳入」之測驗題目、學生作答、作答補充／批改指引，以及「課程內容（RAG 檢索）」，評估學生答案是否正確。
        【quiz_content 測驗題目】
        {qc_disp}
        【quiz_answer 學生作答】
        {qa_disp}
        【answer_user_prompt_text 作答補充／批改指引】
        {aup_disp}
        【課程內容（RAG 檢索）】
        {context_text}
{_GRADE_RUBRIC}
    """

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    normalize_grading_llm_json(llm_json)
    return llm_raw, llm_json


# ---------------------------------------------------------------------------
# 通用背景評分入口
# ---------------------------------------------------------------------------

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

def insert_answer_table_row(table: str, id_column: str, row: dict[str, Any]) -> tuple[str, int] | None:
    """寫入 *Answer 表一列；成功回傳 (id_column 名稱, id)，失敗回傳 None。"""
    try:
        supabase = get_supabase()
        ins = supabase.table(table).insert(row).execute()
        if ins.data and len(ins.data) > 0:
            rid = ins.data[0].get(id_column)
            if rid is not None:
                return (id_column, int(rid))
        _logger.warning(
            "%s insert 成功但未回傳列（無 %s）；若使用 anon key 可能被 RLS 擋",
            table,
            id_column,
        )
    except Exception as e:
        _logger.warning("%s insert 失敗: %s", table, e, exc_info=True)
    return None


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
    """answer_critique JSON 用：作答欄位名（預設 answer_content，與 public.Rag_Quiz／Exam_Quiz 一致）、quiz_grade、quiz_grade_metadata。"""
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
    """更新 public.Rag_Quiz 的題幹（選填）、作答與評分欄位；成功回傳 ("rag_quiz_id", id)。"""
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
    if qc_persist:
        row = {"quiz_content": qc_persist, **row}
    try:
        supabase = get_supabase()
        try:
            supabase.table("Rag_Quiz").update(row).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        except Exception as first_err:
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
    """更新 public.Exam_Quiz 的作答與評分欄位（answer_content, answer_critique）；成功回傳 ("exam_quiz_id", id)。"""
    if exam_quiz_id <= 0:
        return None
    grade = clamp_quiz_grade(result_dict.get("quiz_grade", result_dict.get("score", 0)))
    comments = quiz_comments_from_llm_json(result_dict)
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
