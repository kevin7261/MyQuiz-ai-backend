"""
評分 API：傳入 rag_tab_id、rag_name，程式依 {rag_name}_rag 查找 RAG ZIP，以 RAG 檢索講義後由 GPT-4o 評分。
非同步：POST 回傳 202 + job_id，背景執行評分；前端以 GET /rag/quiz-grade-result/{job_id} 輪詢結果。
"""

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_utc_iso() -> str:
    """回傳目前 UTC 時間的 ISO 字串，供 Rag 表 updated_at 使用。"""
    return datetime.now(timezone.utc).isoformat()

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from utils.rag import process_zip_to_docs
from utils.storage import get_zip_path
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/rag", tags=["rag"])


class GenerateQuizRequest(BaseModel):
    """POST /rag/generate-quiz 請求 body。"""

    llm_api_key: str = Field(..., description="OpenAI API Key")
    rag_id: int = Field(0, description="Rag 表主鍵 rag_id；程式依該筆 rag_metadata.outputs 查找 RAG ZIP")
    rag_tab_id: int = Field(0, description="選填，Rag 表 rag_tab_id（來源 upload 識別）")
    quiz_level: int = Field(0, description="難度等級，會寫入 Rag_Quiz 表 quiz_level")


class QuizGradeRequest(BaseModel):
    """POST /rag/quiz-grade 請求 body。"""

    llm_api_key: str = Field(..., description="OpenAI API Key")
    rag_id: str = Field("", description="Rag 表主鍵 rag_id（字串，會轉成數字查詢）")
    rag_tab_id: str = Field("", description="選填，Rag 表 rag_tab_id（來源識別）")
    rag_quiz_id: str = Field("", description="選填，寫入 Rag_Answer 表 rag_quiz_id")
    quiz_content: str = Field(..., description="測驗題目內容（與 Rag_Quiz 表 quiz_content 一致）")
    answer: str = Field(..., description="學生回答")


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
    quiz_content: str,
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
    docs = retriever.invoke(quiz_content)
    context_text = "\n\n".join([d.page_content for d in docs])

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
    quiz_content: str,
    student_answer: str,
    *,
    rag_tab_id: str = "",
    person_id: str = "",
    rag_id: int = 0,
    rag_quiz_id: int = 0,
) -> None:
    """背景執行評分，結果寫入 _grade_job_results[job_id]，並寫入 public.Rag_Answer 表。"""
    qtype = "short_answer"
    try:
        result = _run_grade_job(work_dir, api_key, quiz_content, student_answer, qtype)
        result_dict = result.model_dump()
        rag_answer_id = None
        try:
            supabase = get_supabase()
            # 完整批改結果以 JSON 字串存入 answer_feedback_metadata，便於前端解析顯示
            answer_feedback_json = json.dumps(result_dict, ensure_ascii=False)
            answer_row = {
                "rag_id": rag_id,
                "rag_tab_id": rag_tab_id or "",
                "rag_quiz_id": rag_quiz_id,
                "person_id": person_id or "",
                "student_answer": student_answer or "",
                "answer_grade": result.score,
                "answer_feedback_metadata": answer_feedback_json,
                "answer_metadata": result_dict,
            }
            ins = supabase.table("Rag_Answer").insert(answer_row).execute()
            if ins.data and len(ins.data) > 0:
                rag_answer_id = ins.data[0].get("rag_answer_id")
        except Exception:
            pass  # 不因寫入 Rag_Answer 失敗而影響輪詢結果
        if rag_answer_id is not None:
            result_dict["rag_answer_id"] = rag_answer_id
        _grade_job_results[job_id] = {
            "status": "ready",
            "result": result_dict,
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


def _rag_stem_from_rag_id(supabase, rag_id: int) -> tuple[dict, str, str]:
    """由 rag_id 查 Rag 表，回傳 (row, stem, rag_zip_tab_id)。stem 取自 rag_metadata.outputs 第一筆的 rag_tab_id；若無則 raise HTTPException。"""
    rag_rows = supabase.table("Rag").select("rag_tab_id, system_prompt_instruction, person_id, rag_id, rag_metadata").eq("rag_id", rag_id).eq("deleted", False).execute()
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    row = rag_rows.data[0]
    meta = row.get("rag_metadata")
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")
    stem = (outputs[0].get("rag_tab_id") or "").strip() if isinstance(outputs[0], dict) else ""
    if not stem:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少 rag_tab_id")
    rag_zip_tab_id = f"{stem}_rag"
    return row, stem, rag_zip_tab_id


@router.post("/generate-quiz")
def generate_quiz_api(body: GenerateQuizRequest):
    """
    傳入 llm_api_key、rag_id（Rag 表主鍵）、rag_tab_id（選填）、quiz_level。
    程式依 rag_id 對應的 rag_metadata.outputs 查找 RAG ZIP 出題；system_prompt_instruction 由 Rag 表取得。
    出題成功後寫入 public.Rag_Quiz 表；回傳 JSON 含 quiz_content, quiz_hint, reference_answer、rag_quiz_id 等。
    """
    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 llm_api_key")
    if not body.rag_id:
        raise HTTPException(status_code=400, detail="請傳入 rag_id")

    supabase = get_supabase()
    row, stem, rag_zip_tab_id = _rag_stem_from_rag_id(supabase, body.rag_id)
    source_rag_tab_id = (row.get("rag_tab_id") or "").strip()
    system_prompt_instruction = (row.get("system_prompt_instruction") or "").strip()
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="該筆 Rag 的 system_prompt_instruction 未設定，請在 build-rag-zip 傳入出題系統指令")

    # 更新該筆 RAG ZIP 對應的 Rag 列（以 rag_zip_tab_id 識別）的 llm_api_key
    try:
        supabase.table("Rag").update({"llm_api_key": (api_key or "").strip() or None, "updated_at": _now_utc_iso()}).eq("rag_tab_id", rag_zip_tab_id).execute()
    except Exception:
        pass
    path = get_zip_path(rag_zip_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={body.rag_id}（rag_tab_id={rag_zip_tab_id}）")

    try:
        from utils.quiz_gen import generate_quiz
        result = generate_quiz(
            path,
            api_key=api_key,
            quiz_level=body.quiz_level,
            system_prompt_instruction=system_prompt_instruction,
        )
        result["system_prompt_instruction"] = system_prompt_instruction
        result["quiz_level"] = body.quiz_level
        result["rag_output"] = {
            "rag_tab_id": stem,
            "rag_name": stem,
            "filename": f"{stem}.zip",
        }
        # 寫入 public.Rag_Quiz 表（須帶入 rag_id、rag_tab_id 為來源 upload 的 rag_tab_id）
        rag_id = int(row.get("rag_id") or 0) if isinstance(row, dict) else 0
        quiz_row: dict[str, Any] = {
            "rag_id": rag_id,
            "rag_tab_id": source_rag_tab_id,
            "person_id": row.get("person_id") or "",
            "quiz_level": body.quiz_level,
            "quiz_content": result.get("quiz_content") or "",
            "quiz_hint": result.get("quiz_hint") or "",
            "reference_answer": result.get("reference_answer") or "",
        }
        # 完整 API 回傳內容寫入 quiz_metadata（與 file_metadata、rag_metadata 模式一致）
        quiz_row["quiz_metadata"] = result
        try:
            quiz_resp = supabase.table("Rag_Quiz").insert(quiz_row).execute()
            if quiz_resp.data and len(quiz_resp.data) > 0:
                result["rag_quiz_id"] = quiz_resp.data[0].get("rag_quiz_id")
                # 以含 rag_quiz_id 的完整回傳更新 quiz_metadata，與實際 API 回傳一致
                supabase.table("Rag_Quiz").update({"quiz_metadata": result}).eq("rag_quiz_id", result["rag_quiz_id"]).eq("rag_tab_id", source_rag_tab_id).execute()
        except Exception:
            pass  # 不因寫入 Rag_Quiz 失敗而影響回傳出題結果
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz-grade")
async def grade_submission(background_tasks: BackgroundTasks, body: QuizGradeRequest):
    """
    傳入 llm_api_key、rag_id（字串）、rag_tab_id（選填）、rag_quiz_id、quiz_content、answer。
    程式依 rag_id 查 Rag 並依 rag_metadata.outputs 查找 RAG ZIP 評分。驗證後回傳 202 與 job_id；背景寫入 public.Rag_Answer。輪詢 GET /rag/quiz-grade-result/{job_id}。
    """
    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "請傳入 llm_api_key"})
    rag_id_str = (body.rag_id or "").strip()
    if not rag_id_str:
        return JSONResponse(status_code=400, content={"error": "請傳入 rag_id"})
    try:
        rag_id_int = int(rag_id_str)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "rag_id 須為數字字串"})

    supabase = get_supabase()
    try:
        row, stem, rag_zip_tab_id = _rag_stem_from_rag_id(supabase, rag_id_int)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    source_rag_tab_id = (row.get("rag_tab_id") or "").strip()
    person_id = (row.get("person_id") or "").strip()
    rag_id_for_answer = int(row.get("rag_id") or 0)

    try:
        supabase.table("Rag").update({"llm_api_key": (api_key or "").strip() or None, "updated_at": _now_utc_iso()}).eq("rag_tab_id", rag_zip_tab_id).execute()
    except Exception:
        pass
    rag_zip_path = get_zip_path(rag_zip_tab_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP，請確認 rag_id={rag_id_str}（tab_id={rag_zip_tab_id}）"})

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

    try:
        rag_quiz_id_int = int((body.rag_quiz_id or "").strip()) if (body.rag_quiz_id or "").strip() else 0
    except ValueError:
        rag_quiz_id_int = 0
    job_id = str(uuid.uuid4())
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    quiz_content = body.quiz_content or ""
    answer = body.answer or ""
    background_tasks.add_task(
        _grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        answer,
        rag_tab_id=source_rag_tab_id,
        person_id=person_id,
        rag_id=rag_id_for_answer,
        rag_quiz_id=rag_quiz_id_int,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["rag"])
async def get_grade_result(job_id: str):
    """
    輪詢評分結果。回傳 status: pending | ready | error；ready 時 result 為批改結果（含 rag_answer_id，對應 public.Rag_Answer 表），error 時 error 為錯誤訊息。
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
