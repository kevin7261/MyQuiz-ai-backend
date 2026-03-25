"""
評分 API 模組。
依 rag_id 自 rag_metadata.outputs 取得 repack stem，再以 {stem}_rag 載入 RAG ZIP 檢索講義後由 GPT-4o 評分。
非同步：POST /rag/grade-quiz 或 POST /rag/quiz-grade（同義）回傳 202 + job_id，背景執行評分；前端以 GET /rag/quiz-grade-result/{job_id} 輪詢結果。
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
# 引入 JSONResponse、Response 用於回傳
from fastapi.responses import JSONResponse, Response
# 引入 Pydantic 的 BaseModel、ConfigDict、Field
from pydantic import AliasChoices, BaseModel, Field, field_validator

# LangChain 文字切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# OpenAI 客戶端
from openai import OpenAI

# 依 person_id 從 User 表取得 LLM API Key
from utils.llm_api_key_utils import get_llm_api_key_for_person
# 從 ZIP 載入文件為 Document 列表
from utils.rag import process_zip_to_docs
# 由 rag_id 取得 stem、rag_zip_tab_id
from utils.rag_common import get_rag_stem_from_rag_id
from utils.system_setting_utils import get_course_name_setting_value
# 取得 ZIP 儲存路徑
from utils.storage import get_zip_path
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


def _quiz_comments_from_llm_json(llm_json: dict[str, Any]) -> list[str]:
    """自 LLM JSON 取出 quiz_comments，正規化為字串列表；若無則相容舊鍵 comments。"""
    raw = llm_json.get("quiz_comments")
    if raw is None:
        raw = llm_json.get("comments")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            c = x.get("comment") if x.get("comment") is not None else x.get("criteria")
            if c is not None:
                out.append(str(c))
        elif x is not None:
            out.append(str(x))
    return out


class GenerateQuizRequest(BaseModel):
    """
    POST /rag/create-quiz 請求 body；欄位順序對齊 public.Rag_Quiz 中由客戶端提供的子集：
    rag_id, rag_tab_id, person_id（後端自 Rag 帶入）, unit_name, file_name（後端帶入）, quiz_level, …
    """

    rag_id: int = Field(0, description="Rag 表主鍵 rag_id；程式依該筆 rag_metadata.outputs 查找 RAG ZIP")
    rag_tab_id: str = Field("", description="選填；覆寫寫入 Rag_Quiz.rag_tab_id，預設沿用 Rag 列之 rag_tab_id")
    unit_name: str = Field(
        "",
        description="選填；指定 rag_metadata.outputs 中某一上傳單元（與 POST /rag/build-rag-zip 的 outputs[].unit_name 一致）。未傳或空字串則使用第一筆輸出",
    )
    quiz_level: str = Field("", description="難度／層級（字串），寫入 Rag_Quiz.quiz_level")

    @field_validator("quiz_level", mode="before")
    @classmethod
    def _quiz_level_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class QuizGradeRequest(BaseModel):
    """
    POST /rag/grade-quiz 請求 body。
    寫入 public.Rag_Answer 時對應：rag_id, rag_tab_id, rag_quiz_id, person_id（後端自 Rag 帶入）,
    quiz_answer（與舊欄位 answer 同義）；評分後寫入 quiz_grade、quiz_grade_metadata。
    """

    # Rag 表主鍵（字串，會轉成數字查詢）
    rag_id: str = Field("", description="Rag 表主鍵 rag_id（字串，會轉成數字查詢）")
    # 選填；目前評分路徑仍以 Rag 列之 rag_tab_id 寫入 Rag_Answer（保留欄位供前端一致帶入）
    rag_tab_id: str = Field("", description="選填；Rag_Answer.rag_tab_id 由後端取自 Rag 列")
    # 選填，寫入 Rag_Answer 表 rag_quiz_id
    rag_quiz_id: str = Field("", description="選填，寫入 Rag_Answer.rag_quiz_id")
    # 測驗題目內容（與 Rag_Quiz 表 quiz_content 一致）
    quiz_content: str = Field(..., description="測驗題目內容（與 Rag_Quiz 表 quiz_content 一致）")
    # 學生作答：請求欄位 quiz_answer（相容舊欄位 answer）；寫入 Rag_Answer.quiz_answer
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Answer.quiz_answer）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


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

    # Embeddings 須與建立 RAG ZIP 時一致（utils.rag 使用 text-embedding-3-small）
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
        # 從 ZIP 載入講義（與 utils.rag.process_zip_to_docs 支援的副檔名一致）
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

    course_name = get_course_name_setting_value()
    # 組裝評分 prompt：角色、目標、限制、評分標準、輸出格式、題目、學生答案、講義依據
    prompt = f"""
        你是一位「{course_name}」課程的教授，請批改這道題目**。
        【評分規範】
        跟據「測驗題目」與「課程內容」，評估「學生回答」的內容是否正確。
        測驗題目：{quiz_content}
        學生回答：{quiz_answer}
        課程內容：{context_text}
        【重要限制】
        1. **請務必使用繁體中文 (Traditional Chinese) 撰寫評語（填入 quiz_comments 陣列）。**
        【評分標準】
        0-5分，一定是整數。
        0: 完全錯誤或未作答。
        1: 只有少量內容正確。
        2: 大幅缺漏，只有部分內容正確。
        3: 部分正確，但有大幅缺漏。
        4: 大致正確，略有不足。
        5: 完全正確且完整。
        【輸出 JSON】
        請以 JSON 格式回傳（quiz_grade 必須為 0 到 5 之間的整數，最高 5）：
        {{ "quiz_grade": int,
        "quiz_comments": [] }}
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
    return llm_raw, llm_json


def _run_grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    quiz_answer: str,
    results_store: dict[str, dict[str, Any]],
    insert_answer_fn: Callable[[dict, str], tuple[str, int] | None],
) -> None:
    """
    通用背景評分：執行評分、可選寫入 DB、結果存 results_store。
    insert_answer_fn(result_dict, quiz_answer) 寫入 DB 並回傳 (id_key, id_val) 或 None。
    """
    try:
        _, llm_json = _run_grade_job(work_dir, api_key, quiz_content, quiz_answer)
        result_dict: dict[str, Any] = {
            "quiz_grade": _quiz_grade_from_llm_json(llm_json),
            "quiz_comments": _quiz_comments_from_llm_json(llm_json),
        }
        # 呼叫 insert_answer_fn 寫入 DB，回傳 (id_key, id_val) 或 None
        inserted = insert_answer_fn(result_dict, quiz_answer)
        # 若有寫入成功，將 id 加入 result_dict
        if inserted:
            result_dict[inserted[0]] = inserted[1]
        else:
            _logger.warning(
                "批改 LLM 已完成但寫入答案表失敗或未取得 id（見上一則 *Answer insert 日誌）。常見原因："
                "未設定 SUPABASE_SERVICE_ROLE_KEY 而改用 anon 遭 RLS 擋、欄位名／型別與表不符（quiz_answer、quiz_grade、quiz_grade_metadata）、或外鍵失敗"
            )
        # 將結果存入 results_store，狀態為 ready
        results_store[job_id] = {"status": "ready", "result": result_dict, "error": None}
        _logger.info(
            "批改完成 job_id=%s 回傳結果: %s",
            job_id,
            json.dumps(result_dict, ensure_ascii=False),
        )
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
    """寫入 Rag_Answer／Exam_Answer：quiz_answer、quiz_grade、quiz_grade_metadata（對齊兩表目前 schema）。"""
    grade = _clamp_quiz_grade(
        result_dict.get("quiz_grade", result_dict.get("score", 0))
    )
    return {
        answer_text_column: quiz_answer or "",
        "quiz_grade": grade,
        "quiz_grade_metadata": result_dict,
    }


def _insert_rag_answer(result_dict: dict, quiz_answer: str, *, rag_id: int, rag_tab_id: str, person_id: str, rag_quiz_id: int) -> tuple[str, int] | None:
    """寫入 public.Rag_Answer，回傳 ("rag_answer_id", id) 或 None。rag_quiz_id<=0 時送 0（與 NOT NULL DEFAULT 0 之 schema 一致）。quiz_answer、quiz_grade、quiz_grade_metadata 與 Exam_Answer 欄位對齊。"""
    row = {
        "rag_id": rag_id,
        "rag_tab_id": rag_tab_id or "",
        "rag_quiz_id": rag_quiz_id if rag_quiz_id > 0 else 0,
        "person_id": person_id or "",
        **_answer_row_payload(result_dict, quiz_answer),
    }
    return _insert_answer_table_row("Rag_Answer", "rag_answer_id", row)


def _insert_exam_answer(result_dict: dict, quiz_answer: str, *, exam_id: int, exam_tab_id: str, person_id: str, exam_quiz_id: int) -> tuple[str, int] | None:
    """寫入 public.Exam_Answer，回傳 ("exam_answer_id", id) 或 None。exam_quiz_id<=0 時送 0（與 NOT NULL DEFAULT 0 之 schema 一致）。作答寫入 quiz_answer；分數與 LLM 結果寫入 quiz_grade、quiz_grade_metadata。"""
    row = {
        "exam_id": exam_id,
        "exam_tab_id": exam_tab_id or "",
        "exam_quiz_id": exam_quiz_id if exam_quiz_id > 0 else 0,
        "person_id": person_id or "",
        **_answer_row_payload(result_dict, quiz_answer),
    }
    return _insert_answer_table_row("Exam_Answer", "exam_answer_id", row)


@router.post("/create-quiz", summary="Rag Create Quiz")
def generate_quiz_api(body: GenerateQuizRequest):
    """
    傳入 rag_id（Rag 表主鍵）、rag_tab_id（選填）、quiz_level；可傳 unit_name 指定 outputs 中哪一個上傳單元（與 build-rag-zip 的 outputs[].unit_name 一致），未傳則用第一筆。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_id 對應的 rag_metadata.outputs 查找 RAG ZIP 出題；system_prompt_instruction 由 Rag 表取得。
    出題成功後寫入 public.Rag_Quiz 表；回傳 JSON 含 quiz_content, quiz_hint, quiz_reference_answer、rag_quiz_id 等。
    """
    # 驗證 rag_id 必填
    if not body.rag_id:
        raise HTTPException(status_code=400, detail="請傳入 rag_id")

    # 取得 Supabase 客戶端
    supabase = get_supabase()
    unit_filter = (body.unit_name or "").strip() or None
    # 由 rag_id 取得 Rag 列、stem、rag_zip_tab_id（可依 unit_name 選 outputs 單元）
    row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
        supabase, body.rag_id, include_row=True, unit_name=unit_filter
    )
    # 從 row 取得 person_id 並去除空白
    person_id = (row.get("person_id") or "").strip()
    # 若 person_id 為空，無法取得 LLM API Key
    if not person_id:
        raise HTTPException(
            status_code=400,
            detail="該筆 Rag 的 person_id 為空，無法取得 LLM API Key",
        )
    # 依 person_id 從 User 表取得 LLM API Key
    api_key = get_llm_api_key_for_person(person_id)
    # 若無 API Key，拋出 400
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
        )
    # 寫入 Rag_Quiz.rag_tab_id：可請求覆寫，否則用該筆 Rag 的 rag_tab_id（varchar）
    source_rag_tab_id = (row.get("rag_tab_id") or "").strip()
    override_rag_tab = (body.rag_tab_id or "").strip()
    quiz_rag_tab_id = override_rag_tab if override_rag_tab else source_rag_tab_id
    # 取得出題系統指令
    system_prompt_instruction = (row.get("system_prompt_instruction") or "").strip()
    # 若未設定，拋出 400
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="該筆 Rag 的 system_prompt_instruction 未設定，請在 build-rag-zip 傳入出題系統指令")

    # 取得 RAG ZIP 的檔案路徑（下載至暫存檔）
    path = get_zip_path(rag_zip_tab_id)
    # 若路徑不存在，拋出 404
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={body.rag_id}（rag_tab_id={rag_zip_tab_id}）")

    try:
        # 動態引入 generate_quiz 避免循環 import
        from utils.create_quiz import generate_quiz
        # 呼叫 generate_quiz 產生題目
        result = generate_quiz(
            path,  # RAG ZIP 路徑
            api_key=api_key,  # LLM API Key
            quiz_level=body.quiz_level,  # 難度等級
            system_prompt_instruction=system_prompt_instruction,  # 出題指令
        )
        # 將 system_prompt_instruction 加入 result
        result["system_prompt_instruction"] = system_prompt_instruction
        # 將 quiz_level 加入 result
        result["quiz_level"] = body.quiz_level
        file_name = f"{stem}.zip"
        # 加入 rag_output 供前端參考
        result["rag_output"] = {
            "rag_tab_id": stem,  # repack stem（與 rag_metadata.outputs[].unit_name 同義）
            "unit_name": stem,
            "filename": file_name,
        }
        # 取得 rag_id 用於寫入 Rag_Quiz
        rag_id = int(row.get("rag_id") or 0) if isinstance(row, dict) else 0
        # 組裝 Rag_Quiz 表要寫入的列（鍵順序同 public.Rag_Quiz，不含 rag_quiz_id / 時間戳）
        quiz_row: dict[str, Any] = {
            "rag_id": rag_id,
            "rag_tab_id": quiz_rag_tab_id,
            "person_id": (row.get("person_id") or "").strip(),
            "unit_name": stem,
            "file_name": file_name,
            "quiz_level": body.quiz_level,
            "quiz_content": result.get("quiz_content") or "",
            "quiz_hint": result.get("quiz_hint") or "",
            "quiz_answer_reference": result.get("quiz_reference_answer") or "",
            "quiz_metadata": result,
        }
        try:
            # 執行 insert 寫入 Rag_Quiz
            quiz_resp = supabase.table("Rag_Quiz").insert(quiz_row).execute()
            # 若有新增成功
            if quiz_resp.data and len(quiz_resp.data) > 0:
                # 將 rag_quiz_id 加入 result
                result["rag_quiz_id"] = quiz_resp.data[0].get("rag_quiz_id")
                # 更新 quiz_metadata 為含 rag_quiz_id 的完整 result
                supabase.table("Rag_Quiz").update({"quiz_metadata": result}).eq("rag_quiz_id", result["rag_quiz_id"]).eq("rag_id", rag_id).eq("rag_tab_id", quiz_rag_tab_id).execute()
        except Exception:
            pass  # 不因寫入 Rag_Quiz 失敗而影響回傳出題結果
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


@router.post("/grade-quiz", summary="Rag Grade Quiz")
@router.post("/quiz-grade", summary="Rag Grade Quiz (alias: /quiz-grade)")
async def grade_submission(background_tasks: BackgroundTasks, body: QuizGradeRequest):
    """
    傳入 rag_id（字串）、rag_tab_id（選填）、rag_quiz_id、quiz_content、quiz_answer。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_id 查 Rag 並依 rag_metadata.outputs 查找 RAG ZIP 評分。驗證後回傳 202 與 job_id；背景寫入 public.Rag_Answer。輪詢 GET /rag/quiz-grade-result/{job_id}，ready 時 result 僅含 quiz_grade、quiz_comments（與 LLM 約定之 JSON）及 rag_answer_id。
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
    # 取得 rag_id 用於寫入 Rag_Answer
    rag_id_for_answer = int(row.get("rag_id") or 0)
    if not person_id:
        return JSONResponse(
            status_code=400,
            content={"error": "該筆 Rag 的 person_id 為空，無法取得 LLM API Key"},
        )
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
    work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_grade_"))
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

    try:
        # 解析 rag_quiz_id，若為空則 0
        rag_quiz_id_int = int((body.rag_quiz_id or "").strip()) if (body.rag_quiz_id or "").strip() else 0
    except ValueError:
        rag_quiz_id_int = 0
    # 產生唯一 job_id
    job_id = str(uuid.uuid4())
    # 初始化 job 狀態為 pending
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    # 建立 insert_fn，用於背景任務寫入 Rag_Answer
    insert_fn = lambda rd, qa: _insert_rag_answer(rd, qa, rag_id=rag_id_for_answer, rag_tab_id=source_rag_tab_id, person_id=person_id, rag_quiz_id=rag_quiz_id_int)
    # 加入背景任務
    background_tasks.add_task(
        _run_grade_job_background,  # 背景任務函數
        job_id,  # job 識別
        work_dir,  # 暫存工作目錄
        api_key,  # LLM API Key
        body.quiz_content or "",  # 題目內容
        body.quiz_answer or "",
        _grade_job_results,  # 結果存放的 dict
        insert_fn,  # 寫入 Rag_Answer 的函數
    )
    # 回傳 202 與 job_id，供前端輪詢
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["rag"])
async def get_grade_result(job_id: str):  # 路徑參數 job_id
    """
    輪詢評分結果。回傳 status: pending | ready | error；
    ready 時 result 為 quiz_grade、quiz_comments（與評分 prompt 之 JSON）及 rag_answer_id；error 時 error 為錯誤訊息。
    此端點刻意保持輕量（僅記憶體查表），以減少代理逾時 502。
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
    # 回傳 status、result、error
    return {
        "status": data["status"],  # pending | ready | error
        "result": data.get("result"),  # 批改結果（ready 時）
        "error": data.get("error"),  # 錯誤訊息（error 時）
    }
