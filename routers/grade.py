"""
評分 API 模組。
依 rag_id 自 rag_metadata.outputs 取得 repack stem，再以 {stem}_rag 載入 RAG ZIP 檢索講義後由 GPT-4o 評分。
非同步：POST /rag/tab/unit/quiz/llm-grade 回傳 202 + job_id，背景執行評分並更新 public.Rag_Quiz 之 answer_* 欄位；前端以 GET /rag/tab/unit/quiz/grade-result/{job_id} 輪詢（寫入失敗時 status 為 error）。POST /rag/tab/unit/quiz/for-exam 將 Rag_Quiz.for_exam 設為 true。列出 for_exam 之 Rag_Unit／Rag_Quiz 請用 GET /exam/rag-for-exams（exam 模組）。RAG+LLM 出題為 POST /rag/tab/unit/quiz/llm-generate；純建 Rag_Quiz 列為 POST /rag/tab/unit/quiz/create（見 zip 路由）。
"""

# 引入 json 用於解析 GPT 回傳與序列化
import json
# 引入 logging 用於終端機輸出批改結果（等同開發時 console 可見）
import logging
# 引入 os 用於 os.walk
import os
# 引入 shutil 用於刪除暫存目錄
import shutil
# 引入 tempfile 用於建立暫存工作目錄
import tempfile
# 引入 uuid 用於產生 job_id
import uuid
# 引入 zipfile 用於解壓 ZIP
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path
# 引入 Any、Callable 型別
from typing import Any, Callable

# 引入 FastAPI 的 APIRouter、BackgroundTasks、HTTPException
from fastapi import APIRouter, BackgroundTasks, HTTPException

from dependencies.person_id import PersonId
# 引入 JSONResponse、Response 用於回傳
from fastapi.responses import JSONResponse, Response
# 引入 Pydantic 的 BaseModel、ConfigDict、Field
from pydantic import AliasChoices, BaseModel, Field

# LangChain 文字切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# OpenAI 客戶端
from openai import OpenAI

from utils.datetime_utils import now_taipei_iso
from utils.course_name_utils import get_course_name_for_prompt
# 依 person_id 從 User 表取得 LLM API Key
from utils.llm_api_key_utils import get_llm_api_key_for_person
# 從 ZIP 載入文件為 Document 列表
from utils.rag_faiss_zip import process_zip_to_docs
# 由 rag_id 取得 stem、rag_zip_tab_id
from utils.rag_stem_utils import get_rag_stem_from_rag_id
# 取得 ZIP 儲存路徑
from utils.zip_storage import get_zip_path
from utils.json_utils import to_json_safe
# Supabase 客戶端
from utils.supabase_client import get_supabase

# 建立路由，前綴 /rag，標籤 rag
router = APIRouter(prefix="/rag", tags=["rag"])

# 批改背景任務完成時寫入日誌（uvicorn 預設會顯示 INFO）
_logger = logging.getLogger(__name__)


def _clamp_quiz_grade(v: Any) -> int:
    """將 quiz_grade 化為 0～5 的整數（滿分固定為 5）。"""
    if v is None:
        return 0
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, n))


def _quiz_grade_from_llm_json(llm_json: dict[str, Any]) -> int:
    """自 LLM JSON 取出分數；優先 quiz_grade，若無則相容舊鍵 score。"""
    v = llm_json.get("quiz_grade")
    if v is None:
        v = llm_json.get("score")
    return _clamp_quiz_grade(v)


def _normalize_grading_llm_json(llm_json: dict[str, Any]) -> None:
    """將舊鍵 comments 併入 quiz_comments 後移除 comments（與 API 欄位名一致）。"""
    if "quiz_comments" not in llm_json and "comments" in llm_json:
        llm_json["quiz_comments"] = llm_json.pop("comments")


def _quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    """自 LLM JSON 取出 quiz_comments，正規化為字串列表。物件元素優先讀 quiz_comment，其次 comment、criteria。"""
    raw = llm_json.get("quiz_comments")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            c = x.get("quiz_comment")
            if c is None:
                c = x.get("comment")
            if c is None:
                c = x.get("criteria")
            if c is not None:
                out.append(str(c))
        elif x is not None:
            out.append(str(x))
    return out


def _quiz_grade_from_answer_critique(critique_raw: Any) -> int | None:
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


def _critique_stored_grade_matches(critique_raw: Any, expected: int) -> bool:
    """answer_critique 內 quiz_grade 是否與預期一致。"""
    g = _quiz_grade_from_answer_critique(critique_raw)
    return g is not None and g == int(expected)


class GenerateQuizRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate；請求 body 僅含 rag_quiz_id、quiz_name、quiz_user_prompt_text。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="使用者出題補充（可空）；併入送 LLM 之 user 訊息並寫入 Rag_Quiz",
    )


class QuizGradeRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade 請求 body。
    欄位順序對齊 public.Rag_Quiz 中實際更新之列：rag_quiz_id、rag_tab_id、quiz_content、answer_user_prompt_text、answer（→answer_content）。
    rag_id 用於載入 RAG ZIP（非 Rag_Quiz 欄位），置於末位。
    """

    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
    rag_tab_id: str = Field("", description="選填；後端以 Rag.rag_tab_id 為準")
    quiz_content: str = Field(..., description="測驗題目內容（與 Rag_Quiz.quiz_content 一致，供 RAG 檢索與批改）")
    answer_user_prompt_text: str = Field(
        "",
        description="作答補充／批改指引（可空）；寫入 Rag_Quiz.answer_user_prompt_text 並供評分 prompt 參考",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Quiz.answer_content）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )
    rag_id: str = Field("", description="Rag 表主鍵 rag_id（字串）；用於解析 RAG ZIP，非 Rag_Quiz 欄位")


class RagQuizForExamRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/for-exam：欄位順序對齊 Rag_Quiz（主鍵與關聯欄）。
    以 **rag_quiz_id** 更新 Rag_Quiz.for_exam = true；若一併傳入 `rag_tab_id`／`rag_unit_id`（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_tab_id: str = Field("", description="選填；與資料列 rag_tab_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")


# 非同步評分結果暫存：job_id -> {"status": "pending"|"ready"|"error", "result": dict|None, "error": str|None}
_grade_job_results: dict[str, dict[str, Any]] = {}


def _cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄。"""
    # 若 work_dir 存在且為有效路徑，則遞迴刪除
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_grade_job(
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    answer_user_prompt_text: str = "",
    *,
    exam_quiz_id: int | None = None,
    rag_quiz_id: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """在給定的 work_dir（已含 ref.zip）執行 RAG + GPT 評分。回傳 (LLM 訊息原文, 解析後 JSON 物件)。"""
    # 工作目錄中的 ZIP 路徑
    zip_source_path = work_dir / "ref.zip"
    # 解壓目錄路徑
    extract_folder = work_dir / "extract"
    # 建立 extract 目錄（若已存在不報錯）
    extract_folder.mkdir(parents=True, exist_ok=True)

    # 若 zip_source_path 不是有效的 ZIP 檔，拋出 ValueError
    if not zipfile.is_zipfile(zip_source_path):
        raise ValueError("無效的 ZIP 檔")

    # 以讀取模式開啟 ZIP 檔
    with zipfile.ZipFile(zip_source_path, "r") as zip_ref:
        # 將 ZIP 全部解壓到 extract_folder
        zip_ref.extractall(extract_folder)

    # 判斷是否為 RAG ZIP（含 FAISS 向量庫）
    is_rag_db = False
    # 存放 FAISS 向量庫所在目錄
    db_folder = None
    # 遍歷 extract_folder 下所有子目錄，尋找 index.faiss 與 index.pkl
    for root, _, files in os.walk(extract_folder):
        # 若當前目錄含 FAISS 索引檔，表示為 RAG 向量庫
        if "index.faiss" in files and "index.pkl" in files:
            is_rag_db = True
            db_folder = root
            break

    # Embeddings 須與建立 RAG ZIP 時一致（utils.rag_faiss_zip 使用 text-embedding-3-small）
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 指定 embedding 模型
        api_key=api_key,  # 傳入 OpenAI API Key
    )

    # 若為 RAG ZIP，直接載入既有 FAISS
    if is_rag_db:
        # 從 db_folder 載入 FAISS 向量庫（allow_dangerous_deserialization 因需載入 pkl）
        vectorstore = FAISS.load_local(
            db_folder,  # FAISS 索引所在目錄
            embeddings,  # 用於查詢的 embeddings
            allow_dangerous_deserialization=True,  # 允許反序列化 pkl
        )
    # 否則為一般講義 ZIP，需先處理文件再建向量庫
    else:
        # 從 ZIP 載入講義（與 utils.rag_faiss_zip.process_zip_to_docs 支援的副檔名一致）
        all_documents = process_zip_to_docs(zip_source_path, extract_folder)
        # 若無任何文件，拋出 ValueError
        if not all_documents:
            raise ValueError("ZIP 內無支援的講義文件")
        # 建立遞迴文字切分器，chunk 1000 字、overlap 200 字
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每個 chunk 最大 1000 字元
            chunk_overlap=200,  # chunk 間重疊 200 字元
        )
        # 將文件切分為多個 chunk
        split_docs = text_splitter.split_documents(all_documents)
        # 從切分後的 documents 建立 FAISS 向量庫
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 將 vectorstore 轉為 Retriever，設定取前 5 筆
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # 以 quiz_content 作為查詢，取得最相關的文件
    docs = retriever.invoke(quiz_content)
    # 將每份文件的 page_content 以雙換行連接成 context_text
    context_text = "\n\n".join([d.page_content for d in docs])

    course_name = get_course_name_for_prompt()
    qc_disp = (quiz_content or "").strip() or "（未提供）"
    qa_disp = (quiz_answer or "").strip() or "（未提供）"
    aup_disp = (answer_user_prompt_text or "").strip() or "（未提供）"
    id_lines: list[str] = []
    if exam_quiz_id is not None and exam_quiz_id > 0:
        id_lines.append(f"        【exam_quiz_id】{exam_quiz_id}")
    if rag_quiz_id is not None and rag_quiz_id > 0:
        id_lines.append(f"        【rag_quiz_id】{rag_quiz_id}")
    id_block = ("\n" + "\n".join(id_lines) + "\n") if id_lines else ""

    # 組裝評分 prompt：API 傳入之題幹／作答／批改指引與 RAG 課程內容
    prompt = f"""
        你是一位「{course_name}」課程的教授，請批改這道題目。
        {id_block}        【評分規範】
        請依下列「API 傳入」之測驗題目、學生作答、作答補充／批改指引，以及「課程內容（RAG 檢索）」，評估學生答案是否正確。
        【quiz_content 測驗題目】
        {qc_disp}
        【quiz_answer 學生作答】
        {qa_disp}
        【answer_user_prompt_text 作答補充／批改指引】
        {aup_disp}
        【課程內容（RAG 檢索）】
        {context_text}
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
        {{ "quiz_grade": int,
        "quiz_comments": str[] }}
    """

    # 建立 OpenAI 客戶端
    client = OpenAI(api_key=api_key)
    # 呼叫 Chat Completions API，強制回傳 JSON
    response = client.chat.completions.create(
        model="gpt-4o",  # 使用 GPT-4o 模型
        messages=[{"role": "user", "content": prompt}],  # 單一 user 訊息
        response_format={"type": "json_object"},  # 強制 JSON 格式
        temperature=0.3,  # 較低溫度以保持評分穩定
    )

    llm_raw = response.choices[0].message.content or ""
    try:
        llm_json = json.loads(llm_raw)
    except json.JSONDecodeError:
        llm_json = {}
    if not isinstance(llm_json, dict):
        llm_json = {}
    _normalize_grading_llm_json(llm_json)
    return llm_raw, llm_json


def _run_grade_job_background(
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
) -> None:
    """
    通用背景評分：執行評分、可選寫入 DB、結果存 results_store。
    insert_answer_fn(result_dict, quiz_answer) 寫入 DB 並回傳 (id_key, id_val) 或 None。
    """
    try:
        _, llm_json = _run_grade_job(
            work_dir,
            api_key,
            quiz_content,
            quiz_answer,
            answer_user_prompt_text,
            exam_quiz_id=exam_quiz_id,
            rag_quiz_id=rag_quiz_id,
        )
        result_dict: dict[str, Any] = {
            "quiz_grade": _quiz_grade_from_llm_json(llm_json),
            "quiz_comments": _quiz_comments_from_llm_json(llm_json),
        }
        # 呼叫 insert_answer_fn 寫入 DB，回傳 (id_key, id_val) 或 None
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
            _logger.warning(
                "批改 LLM 已完成但寫入答案表失敗 job_id=%s：%s",
                job_id,
                err_detail,
            )
            results_store[job_id] = {"status": "error", "result": None, "error": err_detail}
    except Exception as e:
        # 發生異常時，將錯誤訊息存入 results_store
        results_store[job_id] = {"status": "error", "result": None, "error": str(e)}
        _logger.error("批改失敗 job_id=%s: %s", job_id, e, exc_info=True)
    finally:
        # 不論成功或失敗，都清理暫存目錄
        _cleanup_grade_workspace(work_dir)


def _insert_answer_table_row(table: str, id_column: str, row: dict[str, Any]) -> tuple[str, int] | None:
    """寫入 *Answer 表一列；成功回傳 (id_column 名稱, id)，失敗回傳 None（錯誤會寫入日誌）。"""
    try:
        supabase = get_supabase()
        ins = supabase.table(table).insert(row).execute()
        if ins.data and len(ins.data) > 0:
            rid = ins.data[0].get(id_column)
            if rid is not None:
                return (id_column, int(rid))
        _logger.warning(
            "%s insert 成功但未回傳列（無 %s）；若使用 anon key 可能被 RLS 擋或 API 未回傳 representation",
            table,
            id_column,
        )
    except Exception as e:
        _logger.warning("%s insert 失敗: %s", table, e, exc_info=True)
    return None


def _answer_row_payload(
    result_dict: dict,
    quiz_answer: str,
    *,
    answer_text_column: str = "quiz_answer",
) -> dict[str, Any]:
    """Exam_Answer 等：quiz_answer、quiz_grade、quiz_grade_metadata；Rag_Quiz.answer_critique 亦可用同一結構序列化。"""
    grade = _clamp_quiz_grade(
        result_dict.get("quiz_grade", result_dict.get("score", 0))
    )
    return {
        answer_text_column: quiz_answer or "",
        "quiz_grade": grade,
        "quiz_grade_metadata": result_dict,
    }


def _update_rag_quiz_with_grade(
    result_dict: dict,
    quiz_answer: str,
    *,
    rag_quiz_id: int,
    answer_user_prompt_text: str = "",
    quiz_content: str = "",
) -> tuple[str, int] | None:
    """更新 public.Rag_Quiz 的題幹（選填）、作答與評分欄位；成功回傳 (\"rag_quiz_id\", id)。rag_quiz_id<=0 時回傳 None。"""
    if rag_quiz_id <= 0:
        return None
    grade = _clamp_quiz_grade(
        result_dict.get("quiz_grade", result_dict.get("score", 0))
    )
    ts = now_taipei_iso()
    qc_persist = (quiz_content or "").strip()
    # 鍵順序對齊 public.Rag_Quiz：quiz_content（選填）→ answer_* → updated_at
    row: dict[str, Any] = {
        "answer_user_prompt_text": (answer_user_prompt_text or "").strip(),
        "answer_content": quiz_answer or "",
        "answer_critique": json.dumps(_answer_row_payload(result_dict, quiz_answer), ensure_ascii=False),
        "updated_at": ts,
    }
    if qc_persist:
        row = {
            "quiz_content": qc_persist,
            **row,
        }
    try:
        supabase = get_supabase()
        supabase.table("Rag_Quiz").update(row).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
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
                "Rag_Quiz update 後讀不到列（可能 rag_quiz_id=%s 不存在、已刪除或遭 RLS 擋）",
                rag_quiz_id,
            )
            return None
        cr0 = chk.data[0]
        if not _critique_stored_grade_matches(cr0.get("answer_critique"), grade):
            _logger.warning(
                "Rag_Quiz 讀回 answer_critique 內 quiz_grade 與預期 %s 不符（rag_quiz_id=%s），可能更新未套用",
                grade,
                rag_quiz_id,
            )
            return None
        if qc_persist and (cr0.get("quiz_content") or "").strip() != qc_persist:
            _logger.warning(
                "Rag_Quiz 讀回 quiz_content 與預期不符（rag_quiz_id=%s），可能更新未套用",
                rag_quiz_id,
            )
            return None
        return ("rag_quiz_id", rag_quiz_id)
    except Exception as e:
        _logger.warning("Rag_Quiz update 失敗: %s", e, exc_info=True)
    return None


def _update_exam_quiz_with_grade(
    result_dict: dict,
    quiz_answer: str,
    *,
    exam_quiz_id: int,
) -> tuple[str, int] | None:
    """更新 public.Exam_Quiz 的作答與評分欄位（answer_content, answer_critique 含 quiz_grade）；成功回傳 ("exam_quiz_id", id)。"""
    if exam_quiz_id <= 0:
        return None
    grade = _clamp_quiz_grade(result_dict.get("quiz_grade", result_dict.get("score", 0)))
    comments = _quiz_comments_from_llm_json(result_dict)
    critique = json.dumps(
        {"quiz_grade": grade, "quiz_comments": comments},
        ensure_ascii=False,
    )
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
                "Exam_Quiz update 後讀不到列（可能 exam_quiz_id=%s 不存在或遭 RLS 擋）",
                exam_quiz_id,
            )
            return None
        if not _critique_stored_grade_matches(chk.data[0].get("answer_critique"), grade):
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


@router.post("/tab/unit/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def rag_llm_generate_quiz(body: GenerateQuizRequest, caller_person_id: PersonId):
    """
    Body：**`rag_quiz_id`、`quiz_name`、`quiz_user_prompt_text`**（後兩者可空字串）；
    `rag_tab_id`／`rag_unit_id` 由後端依 `rag_quiz_id` 自資料庫帶入；`quiz_name` 空則沿用 stem／單元名。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_quiz_id 對應之 rag_unit_id 查到 Rag_Unit，再回推 Rag 查找 RAG ZIP 出題；
    出題系統指令優先使用 **Rag.system_prompt_instruction**，若為空則使用該單元 **Rag_Unit.quiz_system_prompt_text**（建置時由 **POST /rag/tab/build-rag-zip** 帶入 system_prompt_instruction 寫入）。
    出題成功後**更新** public.Rag_Quiz 錨點列（quiz_name 空字串則沿用 stem／單元名、quiz_*；並清空 answer_* 以免舊作答留存）；回傳 JSON 含 quiz_content, quiz_hint, quiz_reference_answer、quiz_name、rag_quiz_id 等。
    """
    supabase = get_supabase()

    q_sel = (
        supabase.table("Rag_Quiz")
        .select("rag_quiz_id, rag_tab_id, rag_unit_id")
        .eq("rag_quiz_id", body.rag_quiz_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not q_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={body.rag_quiz_id} 的 Rag_Quiz")
    q_row = q_sel.data[0]
    source_rag_unit_id = int(q_row.get("rag_unit_id") or 0)
    if source_rag_unit_id <= 0:
        raise HTTPException(status_code=400, detail="該 rag_quiz_id 對應的 rag_unit_id 無效")

    unit_sel = (
        supabase.table("Rag_Unit")
        .select("rag_unit_id, rag_tab_id, unit_name, quiz_system_prompt_text")
        .eq("rag_unit_id", source_rag_unit_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not unit_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_unit_id={source_rag_unit_id} 的 Rag_Unit")
    unit_row = unit_sel.data[0]
    unit_filter = (unit_row.get("unit_name") or "").strip() or None
    unit_rag_tab_id = (unit_row.get("rag_tab_id") or "").strip()

    source_rag_tab_id = (q_row.get("rag_tab_id") or "").strip()
    if source_rag_tab_id and unit_rag_tab_id and source_rag_tab_id != unit_rag_tab_id:
        raise HTTPException(status_code=400, detail="Rag_Quiz 與 Rag_Unit 的 rag_tab_id 不一致")

    rag_tab_id = source_rag_tab_id or unit_rag_tab_id
    if not rag_tab_id:
        raise HTTPException(status_code=400, detail="無法由 rag_quiz_id 解析 rag_tab_id")

    rag_sel = (
        supabase.table("Rag")
        .select("rag_id")
        .eq("rag_tab_id", rag_tab_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_tab_id={rag_tab_id} 的 Rag")
    rag_id = int(rag_sel.data[0].get("rag_id") or 0)
    if rag_id <= 0:
        raise HTTPException(status_code=400, detail="該 rag_tab_id 對應的 rag_id 無效")

    # 由 rag_id 取得 Rag 列、stem、rag_zip_tab_id（依既有題目的單元名稱對應）
    row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
        supabase, rag_id, include_row=True, unit_name=unit_filter
    )
    # 從 row 取得 person_id 並去除空白
    person_id = (row.get("person_id") or "").strip()
    # 若 person_id 為空，無法取得 LLM API Key
    if not person_id:
        raise HTTPException(
            status_code=400,
            detail="該筆 Rag 的 person_id 為空，無法取得 LLM API Key",
        )
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Rag 出題")
    # 依 person_id 從 User 表取得 LLM API Key
    api_key = get_llm_api_key_for_person(person_id)
    # 若無 API Key，拋出 400
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
        )
    # 出題系統指令：Rag 全 tab 預設，或回退至本單元 Rag_Unit.quiz_system_prompt_text
    system_prompt_instruction = (row.get("system_prompt_instruction") or "").strip()
    if not system_prompt_instruction:
        system_prompt_instruction = (unit_row.get("quiz_system_prompt_text") or "").strip()
    if not system_prompt_instruction:
        raise HTTPException(
            status_code=400,
            detail="出題系統指令未設定：請在 Rag 設定 system_prompt_instruction；若走 ZIP 打包流程可在 POST /rag/tab/build-rag-zip 帶入 system_prompt_instruction（會寫入各 Rag_Unit.quiz_system_prompt_text）",
        )

    # 取得 RAG ZIP 的檔案路徑（下載至暫存檔）
    path = get_zip_path(rag_zip_tab_id)
    # 若路徑不存在，拋出 404
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（rag_tab_id={rag_zip_tab_id}）")

    try:
        # 動態引入 generate_quiz 避免循環 import
        from utils.quiz_generation import generate_quiz
        # 呼叫 generate_quiz 產生題目
        result = generate_quiz(
            path,  # RAG ZIP 路徑
            api_key=api_key,  # LLM API Key
            system_prompt_instruction=system_prompt_instruction,  # 出題指令
            user_instruction=body.quiz_user_prompt_text or "",
        )
        # 將 system_prompt_instruction 加入 result
        result["system_prompt_instruction"] = system_prompt_instruction
        # 加入 rag_output 供前端參考
        result["rag_output"] = {
            "rag_tab_id": stem,  # repack stem（與 rag_metadata.outputs[].unit_name 同義）
            "unit_name": stem,
            "filename": f"{stem}.zip",
        }
        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_reference_answer") or "").strip()
        result["quiz_content"] = qc
        result["quiz_hint"] = qh
        result["quiz_reference_answer"] = qref
        result["rag_quiz_id"] = body.rag_quiz_id
        qup = (body.quiz_user_prompt_text or "").strip()
        qts = now_taipei_iso()
        body_quiz_name = body.quiz_name.strip()
        quiz_name = body_quiz_name or ((stem or "").strip() or (unit_row.get("unit_name") or "").strip() or "")
        result["quiz_name"] = quiz_name
        quiz_update: dict[str, Any] = {
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": qup,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_user_prompt_text": "",
            "answer_content": "",
            "answer_critique": None,
            "updated_at": qts,
        }
        try:
            supabase.table("Rag_Quiz").update(quiz_update).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()
        except Exception as e:
            _logger.error(
                "Rag_Quiz llm-generate 更新失敗 rag_quiz_id=%s: %s",
                body.rag_quiz_id,
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "寫入 Rag_Quiz 失敗。請確認資料表欄位與 API 一致、RLS 是否允許 UPDATE，"
                    "且後端使用 SUPABASE_SERVICE_ROLE_KEY（或具足夠權限的 Secret key）。"
                    f" 原始錯誤：{e}"
                ),
            ) from e

        chk = (
            supabase.table("Rag_Quiz")
            .select("quiz_content, quiz_user_prompt_text")
            .eq("rag_quiz_id", body.rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        row_out = (chk.data or [None])[0]
        if qc and not row_out:
            raise HTTPException(
                status_code=500,
                detail="寫入 Rag_Quiz 後仍讀不到該 rag_quiz_id 列，請檢查主鍵、deleted 狀態或 RLS。",
            )
        if qc and row_out and (row_out.get("quiz_content") or "").strip() != qc:
            _logger.error(
                "Rag_Quiz llm-generate 讀回驗證失敗 rag_quiz_id=%s（預期與實際 quiz_content 不一致）",
                body.rag_quiz_id,
            )
            raise HTTPException(
                status_code=500,
                detail="寫入 Rag_Quiz 未生效（更新後讀回題幹與預期不符）。請檢查 RLS 政策或是否以 anon key 連線導致更新被擋。",
            )
        # 將 result 轉成 JSON bytes，以 UTF-8 編碼
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        # 回傳 JSON Response
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        # ValueError 轉為 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 其他異常轉為 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理從 Supabase Storage 下載的暫存檔
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/tab/unit/quiz/llm-grade", summary="Rag Grade Quiz")
async def grade_submission(background_tasks: BackgroundTasks, body: QuizGradeRequest, caller_person_id: PersonId):
    """
    Body 欄位順序對齊 public.Rag_Quiz（可更新／路由欄）：`rag_quiz_id`、`rag_tab_id`、`quiz_content`、
    `answer_user_prompt_text`、`quiz_answer`（→answer_content）；末欄 `rag_id` 僅供載入 RAG ZIP（非 Rag_Quiz 欄位）。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_id 查 Rag 並依 rag_metadata.outputs 查找 RAG ZIP 評分。**rag_quiz_id 必填**。驗證後回傳 202 與 job_id；背景**更新 public.Rag_Quiz**（answer_*；若 body 帶 quiz_content 則一併寫入題幹）。輪詢 GET /rag/tab/unit/quiz/grade-result/{job_id}，ready 時除 result 外另附 **rag_quiz**（自 DB 讀回之整列）。
    """
    # 取得 rag_id 字串並去除空白
    rag_id_str = (body.rag_id or "").strip()
    # 若為空，回傳 400
    if not rag_id_str:
        return JSONResponse(status_code=400, content={"error": "請傳入 rag_id"})
    try:
        # 嘗試轉成 int
        rag_id_int = int(rag_id_str)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "rag_id 須為數字字串"})

    # 取得 Supabase 客戶端
    supabase = get_supabase()
    try:
        # 由 rag_id 取得 row、stem、rag_zip_tab_id
        row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id_int, include_row=True)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    # 取得來源 rag_tab_id
    source_rag_tab_id = (row.get("rag_tab_id") or "").strip()
    # 取得 person_id
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        return JSONResponse(
            status_code=400,
            content={"error": "該筆 Rag 的 person_id 為空，無法取得 LLM API Key"},
        )
    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Rag 評分"})
    try:
        rag_quiz_id_int = int((body.rag_quiz_id or "").strip()) if (body.rag_quiz_id or "").strip() else 0
    except ValueError:
        rag_quiz_id_int = 0
    if rag_quiz_id_int <= 0:
        return JSONResponse(status_code=400, content={"error": "rag_quiz_id 必填且須為大於 0 的整數（對應 Rag_Quiz 主鍵）"})
    # 依 person_id 取得 LLM API Key
    api_key = get_llm_api_key_for_person(person_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
            },
        )
    # 取得 RAG ZIP 路徑（下載至暫存檔）
    rag_zip_path = get_zip_path(rag_zip_tab_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP，請確認 rag_id={rag_id_str}（tab_id={rag_zip_tab_id}）"})

    # 建立暫存工作目錄
    work_dir = Path(tempfile.mkdtemp(prefix="myquizai_grade_"))
    # 複製後 ZIP 的路徑
    zip_source_path = work_dir / "ref.zip"
    # 解壓目錄路徑
    extract_folder = work_dir / "extract"
    # 建立 extract 目錄
    extract_folder.mkdir(parents=True, exist_ok=True)

    try:
        # 複製 RAG ZIP 到 work_dir，複製完成後立即刪除從 Supabase Storage 下載的暫存檔
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 清理從 Supabase Storage 下載的暫存檔
        try:
            rag_zip_path.unlink(missing_ok=True)
        except Exception:
            pass

    # 產生唯一 job_id
    job_id = str(uuid.uuid4())
    # 初始化 job 狀態為 pending
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    aup = (body.answer_user_prompt_text or "").strip()
    qc_body = (body.quiz_content or "").strip()
    insert_fn = lambda rd, qa: _update_rag_quiz_with_grade(
        rd,
        qa,
        rag_quiz_id=rag_quiz_id_int,
        answer_user_prompt_text=aup,
        quiz_content=qc_body,
    )
    # 加入背景任務
    background_tasks.add_task(
        _run_grade_job_background,  # 背景任務函數
        job_id,  # job 識別
        work_dir,  # 暫存工作目錄
        api_key,  # LLM API Key
        body.quiz_content or "",  # 題目內容
        body.quiz_answer or "",
        _grade_job_results,  # 結果存放的 dict
        insert_fn,  # 更新 Rag_Quiz 評分欄位
        aup,
        rag_quiz_id=rag_quiz_id_int,
    )
    # 回傳 202 與 job_id，供前端輪詢
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.post("/tab/unit/quiz/for-exam", summary="Mark Rag Quiz for exam")
def mark_rag_quiz_for_exam(body: RagQuizForExamRequest, caller_person_id: PersonId):
    """
    將既有 **Rag_Quiz** 的 **for_exam** 設為 **true**。以 `rag_quiz_id` 定位；
    僅 `deleted=false` 且 `person_id` 與呼叫者相符者可更新。
    """
    req_tab = (body.rag_tab_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Rag_Quiz")
            .select("rag_quiz_id, rag_tab_id, rag_unit_id, person_id")
            .eq("rag_quiz_id", body.rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz，或已刪除")

        row0 = sel.data[0]
        pid = (row0.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權更新該 Rag_Quiz")

        if req_tab and (row0.get("rag_tab_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_tab_id 與 rag_quiz_id 對應資料不一致")
        if req_unit > 0 and int(row0.get("rag_unit_id") or 0) != req_unit:
            raise HTTPException(status_code=400, detail="rag_unit_id 與 rag_quiz_id 對應資料不一致")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"for_exam": True, "updated_at": ts}).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", body.rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        row = (read.data or [{}])[0]
        return to_json_safe(
            {
                "rag_quiz_id": row.get("rag_quiz_id"),
                "rag_tab_id": row.get("rag_tab_id"),
                "rag_unit_id": row.get("rag_unit_id"),
                "person_id": row.get("person_id"),
                "quiz_name": row.get("quiz_name"),
                "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
                "quiz_content": row.get("quiz_content"),
                "quiz_hint": row.get("quiz_hint"),
                "quiz_answer_reference": row.get("quiz_answer_reference"),
                "answer_user_prompt_text": row.get("answer_user_prompt_text"),
                "answer_content": row.get("answer_content"),
                "answer_grade": _quiz_grade_from_answer_critique(row.get("answer_critique")),
                "answer_critique": row.get("answer_critique"),
                "for_exam": row.get("for_exam"),
                "deleted": row.get("deleted"),
                "updated_at": row.get("updated_at"),
                "created_at": row.get("created_at"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("POST /rag/tab/unit/quiz/for-exam 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tab/unit/quiz/grade-result/{job_id}", summary="Get Grade Result", tags=["rag"])
async def get_grade_result(job_id: str, _person_id: PersonId):  # 路徑參數 job_id
    """
    輪詢評分結果。回傳 status: pending | ready | error；
    ready 時 result 為 quiz_grade、quiz_comments、rag_quiz_id（另含 rag_answer_id 同值），並自資料庫讀取 **rag_quiz** 整列（與 Rag_Quiz 表一致，供前端確認已持久化）；error 時為 LLM／ZIP 例外，或 LLM 成功但更新 Rag_Quiz 失敗。
    pending／error 時不查資料庫；rag_quiz 讀取失敗時仍回傳 result，rag_quiz 為 null。
    """
    # 若 job_id 不存在於 _grade_job_results（可能服務重啟）
    if job_id not in _grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    # 取得該 job 的資料
    data = _grade_job_results[job_id]
    out: dict[str, Any] = {
        "status": data["status"],  # pending | ready | error
        "result": data.get("result"),  # 批改結果（ready 時）
        "error": data.get("error"),  # 錯誤訊息（error 時）
    }
    rag_quiz_row: dict[str, Any] | None = None
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            rid = res.get("rag_quiz_id")
            if rid is None:
                rid = res.get("rag_answer_id")
            if rid is not None:
                try:
                    rid_int = int(rid)
                    if rid_int > 0:
                        supabase = get_supabase()
                        q = (
                            supabase.table("Rag_Quiz")
                            .select("*")
                            .eq("rag_quiz_id", rid_int)
                            .eq("deleted", False)
                            .limit(1)
                            .execute()
                        )
                        if q.data:
                            rag_quiz_row = to_json_safe(q.data[0])
                except (TypeError, ValueError) as e:
                    _logger.debug("grade-result rag_quiz_id 無效 job_id=%s: %s", job_id, e)
                except Exception as e:
                    _logger.warning("grade-result 讀取 Rag_Quiz 失敗 job_id=%s: %s", job_id, e)
    if data["status"] == "ready":
        out["rag_quiz"] = rag_quiz_row
    return out
