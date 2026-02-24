"""
評分 API：依題目與學生回答，以 RAG 檢索講義後由 GPT-4o 評分。
優先使用上傳的 ZIP，若無則使用伺服器上的 rag_db.zip。

為避免 Render 等平台請求逾時（約 30 秒）導致 502，改為非同步任務：
POST 立即回傳 202 與 job_id，前端輪詢 GET /api/grade_result/{job_id} 取得結果。
"""

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from utils.rag import process_zip_to_docs

router = APIRouter(prefix="/api", tags=["grade"])


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


# 專案根目錄，用於預設 rag_db.zip
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAG_ZIP = _BACKEND_ROOT / "rag_db.zip"

# 非同步任務結果暫存（key: job_id）
_grade_job_store: dict[str, dict[str, Any]] = {}


def _cleanup_grade_workspace(work_dir: Path) -> None:
    """刪除評分過程產生的暫存目錄。"""
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_grading_sync(
    job_id: str,
    file_bytes: Optional[bytes],
    openai_api_key: str,
    question_text: str,
    student_answer: str,
    qtype: str,
) -> None:
    """
    在背景執行評分（同步），完成後寫入 _grade_job_store。
    避免長時間佔用請求，導致 Render 代理逾時 502（且無 CORS 標頭）。
    """
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_grade_"))
        zip_source_path = work_dir / "ref.zip"
        extract_folder = work_dir / "extract"
        extract_folder.mkdir(parents=True, exist_ok=True)

        try:
            if file_bytes is not None:
                zip_source_path.write_bytes(file_bytes)
            else:
                if not _DEFAULT_RAG_ZIP.exists():
                    _cleanup_grade_workspace(work_dir)
                    _grade_job_store[job_id] = {
                        "status": "error",
                        "result": None,
                        "error": "未上傳檔案，且伺服器找不到預設的 rag_db.zip",
                    }
                    return
                shutil.copy(_DEFAULT_RAG_ZIP, zip_source_path)

            if not zipfile.is_zipfile(zip_source_path):
                _cleanup_grade_workspace(work_dir)
                _grade_job_store[job_id] = {"status": "error", "result": None, "error": "無效的 ZIP 檔"}
                return

            with zipfile.ZipFile(zip_source_path, "r") as zip_ref:
                zip_ref.extractall(extract_folder)

            is_rag_db = False
            db_folder = None
            for root, _, files in os.walk(extract_folder):
                if "index.faiss" in files and "index.pkl" in files:
                    is_rag_db = True
                    db_folder = root
                    break

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=openai_api_key,
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
                    _cleanup_grade_workspace(work_dir)
                    _grade_job_store[job_id] = {
                        "status": "error",
                        "result": None,
                        "error": "ZIP 內無支援的講義文件",
                    }
                    return
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

            client = OpenAI(api_key=openai_api_key)
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
            result = GradingResult(
                score=raw.get("score", 0),
                level=raw.get("level", ""),
                rubric=rubric_list,
                strengths=raw.get("strengths", []),
                weaknesses=raw.get("weaknesses", []),
                missing_items=raw.get("missing_items", []),
                action_items=raw.get("action_items", []),
            )
            _grade_job_store[job_id] = {"status": "ready", "result": result.model_dump(), "error": None}
        finally:
            _cleanup_grade_workspace(work_dir)
    except Exception as e:
        _grade_job_store[job_id] = {"status": "error", "result": None, "error": str(e)}


@router.post("/grade_submission")
async def grade_submission_with_upload(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    openai_api_key: str = Form(...),
    question_text: str = Form(...),
    student_answer: str = Form(...),
    qtype: str = Form(...),
):
    """
    【評分 API - 非同步】
    立即回傳 202 與 job_id，避免 Render 等平台請求逾時（約 30 秒）造成 502。
    請以 GET /api/grade_result/{job_id} 輪詢取得批改結果。
    """
    api_key = (openai_api_key or "").strip()
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "請傳入 openai_api_key"},
        )

    file_bytes: Optional[bytes] = None
    if file and file.filename:
        file_bytes = await file.read()

    job_id = uuid.uuid4().hex
    _grade_job_store[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(
        _run_grading_sync,
        job_id,
        file_bytes,
        api_key,
        question_text,
        student_answer,
        qtype,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/grade_result/{job_id}")
async def get_grade_result(job_id: str):
    """
    輪詢批改結果。POST /api/grade_submission 回傳 job_id 後，以此 GET 輪詢直到 status 為 ready 或 error。
    """
    if job_id not in _grade_job_store:
        return JSONResponse(status_code=404, content={"error": "找不到該批改任務"})
    entry = _grade_job_store[job_id]
    status = entry["status"]
    if status == "pending":
        return JSONResponse(status_code=200, content={"status": "pending", "result": None, "error": None})
    if status == "ready":
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "result": entry["result"], "error": None},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "error", "result": None, "error": entry["error"]},
    )
