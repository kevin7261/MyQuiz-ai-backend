"""
評分 API：傳入 file_id、rag_name，程式依 {rag_name}_rag 查找 RAG ZIP，以 RAG 檢索講義後由 GPT-4o 評分。
非同步：POST 回傳 202 + job_id，背景執行評分；前端以 GET /rag/grade_result/{job_id} 輪詢結果。
"""

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from utils.rag import process_zip_to_docs
from utils.storage import get_zip_path
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/rag", tags=["rag"])


class RubricItem(BaseModel):
    """單一評分項目（GPT 可能回傳 criteria、score、comment 等）。"""

    criteria: str = Field(default="", description="評分項目名稱或說明")
    score: Optional[int] = None
    comment: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class GradingResult(BaseModel):
    """批改結果結構化回傳，便於前端分項顯示。"""

    score: int = Field(..., description="總分 (0–10)")
    level: str = Field(..., description="等級，如：優秀、良好、待加強")
    rubric: list[RubricItem] = Field(
        default_factory=list,
        description="各項評分 [概念正確性, 邏輯與解釋, 完整性]",
    )
    strengths: list[str] = Field(default_factory=list, description="優點")
    weaknesses: list[str] = Field(default_factory=list, description="待改進之處")
    missing_items: list[str] = Field(default_factory=list, description="遺漏或未提及的項目")
    action_items: list[str] = Field(default_factory=list, description="建議後續行動")


# 非同步評分結果暫存：job_id -> {"status": "pending"|"ready"|"error", "result": dict|None, "error": str|None}
_grade_job_results: dict[str, dict[str, Any]] = {}


def _cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄。"""
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_grade_job(
    work_dir: Path,
    api_key: str,
    question_text: str,
    student_answer: str,
    qtype: str,
) -> GradingResult:
    """在給定的 work_dir（已含 ref.zip）執行 RAG + GPT 評分，回傳 GradingResult。"""
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

    # 須與建立 RAG ZIP 時一致（utils.rag 使用 text-embedding-3-small）
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    if is_rag_db:
        vectorstore = FAISS.load_local(
            db_folder,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        all_documents = process_zip_to_docs(zip_source_path, extract_folder)
        if not all_documents:
            raise ValueError("ZIP 內無支援的講義文件")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question_text)
    context_text = "\n\n".join([d.page_content for d in docs])

    prompt = f"""你是一位「地理資訊系統與環境資料分析」助教。請批改這道**觀念簡答題**。
                目標：評估學生對「地理資訊系統與環境資料分析」的理解、邏輯推演與解釋清晰度。
                【重要限制】
                1. **請務必使用繁體中文 (Traditional Chinese) 撰寫所有評語、優點、弱點與行動建議。**
                【評分標準】A) 概念正確性 (3分), B) 邏輯與解釋 (4分), C) 完整性 (3分)。
                【輸出 JSON】{{ "score": int, "level": str, "rubric": [], "strengths": [], "weaknesses": [], "missing_items": [], "action_items": [] }}
                [題目] {question_text}
                [學生回答] {student_answer}
                [講義依據] {context_text}
            """

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = json.loads(response.choices[0].message.content)
    rubric_raw = raw.get("rubric", [])
    rubric_list = []
    for item in rubric_raw:
        if isinstance(item, dict):
            rubric_list.append(RubricItem.model_validate(item))
        else:
            rubric_list.append(RubricItem(criteria=str(item)))
    return GradingResult(
        score=raw.get("score", 0),
        level=raw.get("level", ""),
        rubric=rubric_list,
        strengths=raw.get("strengths", []),
        weaknesses=raw.get("weaknesses", []),
        missing_items=raw.get("missing_items", []),
        action_items=raw.get("action_items", []),
    )


def _grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    question_text: str,
    student_answer: str,
    qtype: str,
) -> None:
    """背景執行評分，結果寫入 _grade_job_results[job_id]。"""
    try:
        result = _run_grade_job(work_dir, api_key, question_text, student_answer, qtype)
        _grade_job_results[job_id] = {
            "status": "ready",
            "result": result.model_dump(),
            "error": None,
        }
    except Exception as e:
        _grade_job_results[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e),
        }
    finally:
        _cleanup_grade_workspace(work_dir)


@router.post("/grade_submission")
async def grade_submission(
    background_tasks: BackgroundTasks,
    file_id: str = Form(..., description="upload-zip 回傳的 source file_id"),
    rag_name: str = Form(..., description="rag_list 某一段的 stem，如 220222_220301；程式會以 {rag_name}_rag 查找 RAG ZIP"),
    question_text: str = Form(..., description="題目文字"),
    student_answer: str = Form(..., description="學生回答"),
    qtype: str = Form(..., description="題型，如 short_answer"),
):
    """
    傳入 file_id（upload-zip 的 source file_id）與 rag_name（如 220222_220301），程式自動組出 rag_file_id={rag_name}_rag 並查找 RAG ZIP。
    OpenAI API key 由 Rag 表該 file_id 的 llm_api_key 取得；若為空則回傳 400。
    驗證後立即回傳 202 與 job_id；實際評分在背景執行。前端請以 GET /rag/grade_result/{job_id} 輪詢。
    """
    file_id = (file_id or "").strip()
    if not file_id:
        return JSONResponse(status_code=400, content={"error": "請傳入 file_id"})

    # API key 由 Rag 表該 file_id 的 llm_api_key 取得
    supabase = get_supabase()
    resp = supabase.table("Rag").select("llm_api_key").eq("file_id", file_id).execute()
    row = (resp.data or [None])[0] if resp.data else None
    api_key = (row.get("llm_api_key") or "").strip() if isinstance(row, dict) else ""
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "該 file_id 的 Rag 尚未設定 llm_api_key，請在上傳 ZIP 或「確定修改」處填入 OpenAI API Key"})

    rag_name = (rag_name or "").strip()
    if not rag_name:
        return JSONResponse(status_code=400, content={"error": "請傳入 rag_name（如 220222_220301）"})

    rag_file_id = f"{rag_name}_rag"
    rag_zip_path = get_zip_path(rag_file_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"找不到 RAG ZIP，請確認 file_id={file_id}、rag_name={rag_name}（rag_file_id={rag_file_id}）"},
        )

    work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_grade_"))
    zip_source_path = work_dir / "ref.zip"
    extract_folder = work_dir / "extract"
    extract_folder.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})

    job_id = str(uuid.uuid4())
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(
        _grade_job_background,
        job_id,
        work_dir,
        api_key,
        question_text,
        student_answer,
        qtype,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/grade_result/{job_id}", tags=["rag"])
async def get_grade_result(job_id: str):
    """
    輪詢評分結果。回傳 status: pending | ready | error；ready 時 result 為批改結果，error 時 error 為錯誤訊息。
    此端點刻意保持輕量（僅記憶體查表），以減少代理逾時 502。
    """
    if job_id not in _grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _grade_job_results[job_id]
    return {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
    }
