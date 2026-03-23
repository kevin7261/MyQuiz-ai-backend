"""
評分 API 模組。
傳入 rag_tab_id、rag_name，程式依 {rag_name}_rag 查找 RAG ZIP，以 RAG 檢索講義後由 GPT-4o 評分。
非同步：POST 回傳 202 + job_id，背景執行評分；前端以 GET /rag/quiz-grade-result/{job_id} 輪詢結果。
"""

# 引入 json 用於解析 GPT 回傳與序列化
import json
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
# 引入 Any、Callable、Optional 型別
from typing import Any, Callable, Optional

# 引入 FastAPI 的 APIRouter、BackgroundTasks、HTTPException
from fastapi import APIRouter, BackgroundTasks, HTTPException
# 引入 JSONResponse、Response 用於回傳
from fastapi.responses import JSONResponse, Response
# 引入 Pydantic 的 BaseModel、ConfigDict、Field
from pydantic import BaseModel, ConfigDict, Field

# LangChain 文字切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# OpenAI 客戶端
from openai import OpenAI

# UTC 時間工具（本模組未直接使用，保留供擴充）
from utils.datetime_utils import now_utc_iso
# 依 person_id 從 User 表取得 LLM API Key
from utils.llm_api_key_utils import get_llm_api_key_for_person
# 從 ZIP 載入文件為 Document 列表
from utils.rag import process_zip_to_docs
# 由 rag_id 取得 stem、rag_zip_tab_id
from utils.rag_common import get_rag_stem_from_rag_id
# 取得 ZIP 儲存路徑
from utils.storage import get_zip_path
# Supabase 客戶端
from utils.supabase_client import get_supabase

# 建立路由，前綴 /rag，標籤 rag
router = APIRouter(prefix="/rag", tags=["rag"])


class GenerateQuizRequest(BaseModel):
    """
    POST /rag/generate-quiz 請求 body。
    欄位順序與 Rag_Quiz 表一致：rag_id, rag_tab_id, quiz_level。
    LLM API Key 依 Rag 的 person_id 從 User 表取得。
    """

    # Rag 表主鍵；程式依該筆 rag_metadata.outputs 查找 RAG ZIP
    rag_id: int = Field(0, description="Rag 表主鍵 rag_id；程式依該筆 rag_metadata.outputs 查找 RAG ZIP")
    # 選填，Rag 表 rag_tab_id（來源 upload 識別）
    rag_tab_id: int = Field(0, description="選填，Rag 表 rag_tab_id（來源 upload 識別）")
    # 難度等級，會寫入 Rag_Quiz 表 quiz_level
    quiz_level: int = Field(0, description="難度等級，會寫入 Rag_Quiz 表 quiz_level")


class QuizGradeRequest(BaseModel):
    """
    POST /rag/quiz-grade 請求 body。
    欄位順序與 Rag_Answer 表一致：rag_id, rag_tab_id, rag_quiz_id, quiz_content, answer。
    LLM API Key 依 Rag 的 person_id 從 User 表取得。
    """

    # Rag 表主鍵（字串，會轉成數字查詢）
    rag_id: str = Field("", description="Rag 表主鍵 rag_id（字串，會轉成數字查詢）")
    # 選填，Rag 表 rag_tab_id（來源識別）
    rag_tab_id: str = Field("", description="選填，Rag 表 rag_tab_id（來源識別）")
    # 選填，寫入 Rag_Answer 表 rag_quiz_id
    rag_quiz_id: str = Field("", description="選填，寫入 Rag_Answer 表 rag_quiz_id")
    # 測驗題目內容（與 Rag_Quiz 表 quiz_content 一致）
    quiz_content: str = Field(..., description="測驗題目內容（與 Rag_Quiz 表 quiz_content 一致）")
    # 學生回答內容
    answer: str = Field(..., description="學生回答")


class RubricItem(BaseModel):
    """單一評分項目（GPT 可能回傳 criteria、score、comment 等）。"""

    # 評分項目名稱或說明
    criteria: str = Field(default="", description="評分項目名稱或說明")
    # 該項得分
    score: Optional[int] = None
    # 該項評語
    comment: Optional[str] = None
    # 允許額外欄位（GPT 可能回傳其他欄位）
    model_config = ConfigDict(extra="allow")


class GradingResult(BaseModel):
    """批改結果結構化回傳，便於前端分項顯示。"""

    # 總分（0–10）
    score: int = Field(..., description="總分 (0–10)")
    # 等級，如：優秀、良好、待加強
    level: str = Field(..., description="等級，如：優秀、良好、待加強")
    # 各項評分 [概念正確性, 邏輯與解釋, 完整性]
    rubric: list[RubricItem] = Field(
        default_factory=list,
        description="各項評分 [概念正確性, 邏輯與解釋, 完整性]",
    )
    # 優點列表
    strengths: list[str] = Field(default_factory=list, description="優點")
    # 待改進之處
    weaknesses: list[str] = Field(default_factory=list, description="待改進之處")
    # 遺漏或未提及的項目
    missing_items: list[str] = Field(default_factory=list, description="遺漏或未提及的項目")
    # 建議後續行動
    action_items: list[str] = Field(default_factory=list, description="建議後續行動")


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
    student_answer: str,
    qtype: str,
) -> GradingResult:
    """
    在給定的 work_dir（已含 ref.zip）執行 RAG + GPT 評分，回傳 GradingResult。
    qtype 預留供未來擴充題型（目前皆為 short_answer）。
    """
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

    # 組裝評分 prompt：角色、目標、限制、評分標準、輸出格式、題目、學生答案、講義依據
    prompt = f"""你是一位「地理資訊系統與環境資料分析」助教。請批改這道**觀念簡答題**。
                目標：評估學生對「地理資訊系統與環境資料分析」的理解、邏輯推演與解釋清晰度。
                【重要限制】
                1. **請務必使用繁體中文 (Traditional Chinese) 撰寫所有評語、優點、弱點與行動建議。**
                【評分標準】A) 概念正確性 (3分), B) 邏輯與解釋 (4分), C) 完整性 (3分)。
                【輸出 JSON】{{ "score": int, "level": str, "rubric": [], "strengths": [], "weaknesses": [], "missing_items": [], "action_items": [] }}
                [測驗題目（quiz）] {quiz_content}
                [學生回答] {student_answer}
                [講義依據] {context_text}
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

    # 從 response 取出 content 並解析為 dict
    raw = json.loads(response.choices[0].message.content)
    # 取得 rubric 陣列，若無則為 []
    rubric_raw = raw.get("rubric", [])
    # 用於存放 RubricItem 的列表
    rubric_list = []
    # 遍歷 rubric_raw 每筆，轉成 RubricItem
    for item in rubric_raw:
        # 若為 dict，用 model_validate 轉換
        if isinstance(item, dict):
            rubric_list.append(RubricItem.model_validate(item))
        # 否則以 str(item) 作為 criteria
        else:
            rubric_list.append(RubricItem(criteria=str(item)))
    # 組裝 GradingResult 並回傳
    return GradingResult(
        score=raw.get("score", 0),  # 總分，預設 0
        level=raw.get("level", ""),  # 等級
        rubric=rubric_list,  # 各項評分
        strengths=raw.get("strengths", []),  # 優點
        weaknesses=raw.get("weaknesses", []),  # 弱點
        missing_items=raw.get("missing_items", []),  # 遺漏項目
        action_items=raw.get("action_items", []),  # 建議行動
    )


def _run_grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    student_answer: str,
    results_store: dict[str, dict[str, Any]],
    insert_answer_fn: Callable[[dict, str], tuple[str, int] | None],
) -> None:
    """
    通用背景評分：執行評分、可選寫入 DB、結果存 results_store。
    insert_answer_fn(result_dict, student_answer) 寫入 DB 並回傳 (id_key, id_val) 或 None。
    """
    try:
        # 呼叫 _run_grade_job 執行評分
        result = _run_grade_job(work_dir, api_key, quiz_content, student_answer, "short_answer")
        # 將 Pydantic 模型轉為 dict
        result_dict = result.model_dump()
        # 呼叫 insert_answer_fn 寫入 DB，回傳 (id_key, id_val) 或 None
        inserted = insert_answer_fn(result_dict, student_answer)
        # 若有寫入成功，將 id 加入 result_dict
        if inserted:
            result_dict[inserted[0]] = inserted[1]
        # 將結果存入 results_store，狀態為 ready
        results_store[job_id] = {"status": "ready", "result": result_dict, "error": None}
    except Exception as e:
        # 發生異常時，將錯誤訊息存入 results_store
        results_store[job_id] = {"status": "error", "result": None, "error": str(e)}
    finally:
        # 不論成功或失敗，都清理暫存目錄
        _cleanup_grade_workspace(work_dir)


def _insert_rag_answer(result_dict: dict, student_answer: str, *, rag_id: int, rag_tab_id: str, person_id: str, rag_quiz_id: int) -> tuple[str, int] | None:
    """寫入 Rag_Answer 表，回傳 ("rag_answer_id", id) 或 None。"""
    try:
        # 取得 Supabase 客戶端
        supabase = get_supabase()
        # 將 result_dict 轉成 JSON 字串存入 answer_feedback_metadata
        answer_feedback_json = json.dumps(result_dict, ensure_ascii=False)
        # 組裝要寫入 Rag_Answer 的列
        answer_row = {
            "rag_id": rag_id,  # Rag 主鍵
            "rag_tab_id": rag_tab_id or "",  # Rag tab 識別
            "rag_quiz_id": rag_quiz_id,  # 題目主鍵
            "person_id": person_id or "",  # 使用者識別
            "student_answer": student_answer or "",  # 學生回答內容
            "answer_grade": result_dict.get("score", 0),  # 分數
            "answer_feedback_metadata": answer_feedback_json,  # JSON 字串格式的完整回饋
            "answer_metadata": result_dict,  # dict 格式的完整回饋
        }
        # 執行 insert 並取得回傳
        ins = supabase.table("Rag_Answer").insert(answer_row).execute()
        # 若有新增資料
        if ins.data and len(ins.data) > 0:
            # 取得 rag_answer_id
            rid = ins.data[0].get("rag_answer_id")
            # 若 id 不為 None，回傳 ("rag_answer_id", id)
            if rid is not None:
                return ("rag_answer_id", int(rid))
    except Exception:
        # 靜默忽略任何異常
        pass
    # 寫入失敗或無 id 時回傳 None
    return None


def _insert_exam_answer(result_dict: dict, student_answer: str, *, exam_id: int, exam_tab_id: str, person_id: str, exam_quiz_id: int) -> tuple[str, int] | None:
    """寫入 Exam_Answer 表，回傳 ("exam_answer_id", id) 或 None。"""
    try:
        # 取得 Supabase 客戶端
        supabase = get_supabase()
        # 將 result_dict 轉成 JSON 字串
        answer_feedback_json = json.dumps(result_dict, ensure_ascii=False)
        # 組裝要寫入 Exam_Answer 的列
        answer_row = {
            "exam_id": exam_id,  # Exam 主鍵
            "exam_tab_id": exam_tab_id or "",  # Exam tab 識別
            "exam_quiz_id": exam_quiz_id,  # 題目主鍵
            "person_id": person_id or "",  # 使用者識別
            "student_answer": student_answer or "",  # 學生回答內容
            "answer_grade": result_dict.get("score", 0),  # 分數
            "answer_feedback_metadata": answer_feedback_json,  # JSON 字串格式回饋
            "answer_metadata": result_dict,  # dict 格式回饋
        }
        # 執行 insert
        ins = supabase.table("Exam_Answer").insert(answer_row).execute()
        # 若有新增資料
        if ins.data and len(ins.data) > 0:
            rid = ins.data[0].get("exam_answer_id")
            if rid is not None:
                return ("exam_answer_id", int(rid))
    except Exception:
        pass
    return None


@router.post("/generate-quiz")
def generate_quiz_api(body: GenerateQuizRequest):
    """
    傳入 rag_id（Rag 表主鍵）、rag_tab_id（選填）、quiz_level。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_id 對應的 rag_metadata.outputs 查找 RAG ZIP 出題；system_prompt_instruction 由 Rag 表取得。
    出題成功後寫入 public.Rag_Quiz 表；回傳 JSON 含 quiz_content, quiz_hint, reference_answer、rag_quiz_id 等。
    """
    # 驗證 rag_id 必填
    if not body.rag_id:
        raise HTTPException(status_code=400, detail="請傳入 rag_id")

    # 取得 Supabase 客戶端
    supabase = get_supabase()
    # 由 rag_id 取得 Rag 列、stem、rag_zip_tab_id
    row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, body.rag_id, include_row=True)
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
    # 取得來源 rag_tab_id（用於寫入 Rag_Quiz）
    source_rag_tab_id = (row.get("rag_tab_id") or "").strip()
    # 取得出題系統指令
    system_prompt_instruction = (row.get("system_prompt_instruction") or "").strip()
    # 若未設定，拋出 400
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="該筆 Rag 的 system_prompt_instruction 未設定，請在 build-rag-zip 傳入出題系統指令")

    # 取得 RAG ZIP 的檔案路徑
    path = get_zip_path(rag_zip_tab_id)
    # 若路徑不存在，拋出 404
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={body.rag_id}（rag_tab_id={rag_zip_tab_id}）")

    try:
        # 動態引入 generate_quiz 避免循環 import
        from utils.quiz_gen import generate_quiz
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
        # 加入 rag_output 供前端參考
        result["rag_output"] = {
            "rag_tab_id": stem,  # RAG tab 識別
            "rag_name": stem,  # RAG 名稱
            "filename": f"{stem}.zip",  # 檔名
        }
        # 取得 rag_id 用於寫入 Rag_Quiz
        rag_id = int(row.get("rag_id") or 0) if isinstance(row, dict) else 0
        # 組裝 Rag_Quiz 表要寫入的列
        quiz_row: dict[str, Any] = {
            "rag_id": rag_id,  # Rag 主鍵
            "rag_tab_id": source_rag_tab_id,  # 來源 upload 的 rag_tab_id
            "person_id": row.get("person_id") or "",  # 使用者識別
            "quiz_level": body.quiz_level,  # 難度等級
            "quiz_content": result.get("quiz_content") or "",  # 題目內容
            "quiz_hint": result.get("quiz_hint") or "",  # 提示
            "reference_answer": result.get("reference_answer") or "",  # 參考答案
        }
        # 完整 API 回傳內容寫入 quiz_metadata（與 file_metadata、rag_metadata 模式一致）
        quiz_row["quiz_metadata"] = result
        try:
            # 執行 insert 寫入 Rag_Quiz
            quiz_resp = supabase.table("Rag_Quiz").insert(quiz_row).execute()
            # 若有新增成功
            if quiz_resp.data and len(quiz_resp.data) > 0:
                # 將 rag_quiz_id 加入 result
                result["rag_quiz_id"] = quiz_resp.data[0].get("rag_quiz_id")
                # 更新 quiz_metadata 為含 rag_quiz_id 的完整 result
                supabase.table("Rag_Quiz").update({"quiz_metadata": result}).eq("rag_quiz_id", result["rag_quiz_id"]).eq("rag_tab_id", source_rag_tab_id).execute()
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


@router.post("/quiz-grade")
async def grade_submission(background_tasks: BackgroundTasks, body: QuizGradeRequest):
    """
    傳入 rag_id（字串）、rag_tab_id（選填）、rag_quiz_id、quiz_content、answer。
    LLM API Key 依 Rag 的 person_id 從 User 表取得；請確保該使用者已於個人設定填寫 LLM API Key。
    程式依 rag_id 查 Rag 並依 rag_metadata.outputs 查找 RAG ZIP 評分。驗證後回傳 202 與 job_id；背景寫入 public.Rag_Answer。輪詢 GET /rag/quiz-grade-result/{job_id}。
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
    # 取得 RAG ZIP 路徑
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
        # 複製 RAG ZIP 到 work_dir
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})

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
    insert_fn = lambda rd, sa: _insert_rag_answer(rd, sa, rag_id=rag_id_for_answer, rag_tab_id=source_rag_tab_id, person_id=person_id, rag_quiz_id=rag_quiz_id_int)
    # 加入背景任務
    background_tasks.add_task(
        _run_grade_job_background,  # 背景任務函數
        job_id,  # job 識別
        work_dir,  # 暫存工作目錄
        api_key,  # LLM API Key
        body.quiz_content or "",  # 題目內容
        body.answer or "",  # 學生回答
        _grade_job_results,  # 結果存放的 dict
        insert_fn,  # 寫入 Rag_Answer 的函數
    )
    # 回傳 202 與 job_id，供前端輪詢
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["rag"])
async def get_grade_result(job_id: str):  # 路徑參數 job_id
    """
    輪詢評分結果。回傳 status: pending | ready | error；
    ready 時 result 為批改結果（含 rag_answer_id）；error 時 error 為錯誤訊息。
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
