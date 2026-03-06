"""
Exam API：對應 public.Exam / Exam_Quiz / Exam_Answer 表。
- GET /exam/exams：列出 Exam 表（格式同 GET /rag/rags），每筆含 quizzes（每題帶 answers）與頂層 answers。
- POST /exam/create-exam：建立一筆 Exam 資料。
- POST /exam/generate-quiz：依 exam_tab_id 與 rag_id 查找 RAG ZIP 出題，寫入 Exam_Quiz（exam_id, exam_tab_id）。
- POST /exam/quiz-grade：非同步評分，寫入 Exam_Answer；輪詢 GET /exam/quiz-grade-result/{job_id}。
- POST /exam/delete/{exam_tab_id}：軟刪除該筆 Exam（deleted=true）。
"""

import json
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Path as PathParam, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from utils.storage import generate_tab_id, get_zip_path
from utils.supabase_client import get_supabase

from routers.grade import _run_grade_job, _cleanup_grade_workspace

router = APIRouter(prefix="/exam", tags=["exam"])


# --- GET /exam/exams（格式同 /rag/rags）---

def _exams_table_select(exclude_deleted: bool = True) -> list[dict]:
    """查詢 Exam 表，exclude_deleted=True 時僅回傳 deleted=False。"""
    supabase = get_supabase()
    q = supabase.table("Exam").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    resp = q.execute()
    return resp.data or []


def _quizzes_by_exam_id(exam_ids: list[int]) -> dict[int, list[dict]]:
    """依 exam_id 查詢 Exam_Quiz 表，回傳 exam_id -> list of quiz。"""
    if not exam_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Exam_Quiz").select("*").in_("exam_id", exam_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {eid: [] for eid in exam_ids}
    for row in rows:
        eid = row.get("exam_id")
        if eid is not None:
            try:
                out.setdefault(int(eid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


def _answers_by_exam_id(exam_ids: list[int]) -> dict[int, list[dict]]:
    """依 exam_id 查詢 Exam_Answer 表，回傳 exam_id -> list of answer。"""
    if not exam_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Exam_Answer").select("*").in_("exam_id", exam_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {eid: [] for eid in exam_ids}
    for row in rows:
        eid = row.get("exam_id")
        if eid is not None:
            try:
                out.setdefault(int(eid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


def _quizzes_by_person_id(person_id: str) -> list[dict]:
    """依 person_id 查詢 Exam_Quiz 表，回傳該使用者的所有題目。"""
    pid = (person_id or "").strip()
    if not pid:
        return []
    supabase = get_supabase()
    resp = supabase.table("Exam_Quiz").select("*").eq("person_id", pid).execute()
    return resp.data or []


def _answers_by_exam_quiz_ids(exam_quiz_ids: list[int]) -> dict[int, list[dict]]:
    """依 exam_quiz_id 查詢 Exam_Answer 表，回傳 exam_quiz_id -> list of answer。"""
    if not exam_quiz_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Exam_Answer").select("*").in_("exam_quiz_id", exam_quiz_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {qid: [] for qid in exam_quiz_ids}
    for row in rows:
        qid = row.get("exam_quiz_id")
        if qid is not None:
            try:
                out.setdefault(int(qid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


class ListExamResponse(BaseModel):
    """GET /exam/exams 回應：Exam 表全部資料，每筆另含關聯的 Exam_Quiz（quizzes，每題帶 answers）與頂層 Exam_Answer（answers）。"""
    exams: list[dict]
    count: int


@router.get("/exams", response_model=ListExamResponse)
def list_exams(
    person_id: Optional[str] = Query(None, description="選填，篩選 person_id；未傳則回傳全部"),
):
    """
    列出 Exam 表內容，僅回傳 deleted=False 的資料；每筆 Exam 含表上所有欄位，並帶關聯的 Exam_Quiz（quizzes，每題帶 answers）與頂層 answers。
    格式同 GET /rag/rags。
    """
    try:
        data = _exams_table_select(exclude_deleted=True)
        if person_id is not None and str(person_id).strip():
            pid = str(person_id).strip()
            data = [r for r in data if (r.get("person_id") or "").strip() == pid]
        exam_ids = []
        for row in data:
            eid = row.get("exam_id")
            if eid is not None:
                try:
                    exam_ids.append(int(eid))
                except (TypeError, ValueError):
                    pass
        exam_ids = list(dict.fromkeys(exam_ids))
        quizzes_by_exam = _quizzes_by_exam_id(exam_ids)
        answers_by_exam = _answers_by_exam_id(exam_ids)
        # 依 exam_quiz_id 彙總 answers，供每筆 quiz 帶關聯的 answers
        answers_by_quiz_id: dict[int, list[dict]] = {}
        for eid in exam_ids:
            for a in answers_by_exam.get(eid, []):
                qid = a.get("exam_quiz_id")
                if qid is not None:
                    try:
                        qid_int = int(qid)
                        answers_by_quiz_id.setdefault(qid_int, []).append(a)
                    except (TypeError, ValueError):
                        pass
        for row in data:
            eid = row.get("exam_id")
            eid_int = int(eid) if eid is not None else None
            row_quizzes = quizzes_by_exam.get(eid_int, []) if eid_int is not None else []
            for quiz in row_quizzes:
                qid = quiz.get("exam_quiz_id")
                qid_int = int(qid) if qid is not None else None
                quiz["answers"] = answers_by_quiz_id.get(qid_int, []) if qid_int is not None else []
            row["quizzes"] = row_quizzes
            row["answers"] = answers_by_exam.get(eid_int, []) if eid_int is not None else []
        return ListExamResponse(exams=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateExamRequest(BaseModel):
    """POST /exam/create-exam 請求 body。欄位順序與 Exam 表一致：exam_tab_id, person_id, exam_name。"""

    exam_tab_id: str | None = Field(None, description="選填；未傳則由後端產生（格式同 tab_id）")
    person_id: str = Field("", description="選填，寫入 Exam 表 person_id")
    exam_name: str = Field("", description="測驗名稱，寫入 Exam 表 exam_name")


class ExamGenerateQuizRequest(BaseModel):
    """POST /exam/generate-quiz 請求 body。欄位順序與 Exam_Quiz 表一致：exam_id, exam_tab_id, quiz_level；llm_api_key 為呼叫用。"""

    exam_id: int = Field(0, description="Exam 表主鍵 exam_id")
    exam_tab_id: str | int = Field("", description="create-exam 回傳的 exam_tab_id（Exam 表識別）；與 exam_id 二擇一，可傳字串或 0")
    quiz_level: int = Field(0, description="難度等級，寫入 Exam_Quiz 表 quiz_level")
    llm_api_key: str = Field(..., description="LLM API Key")


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/quiz-grade 請求 body。欄位順序與 Exam_Answer 表一致：exam_id, exam_tab_id, exam_quiz_id, quiz_content, answer；llm_api_key 為呼叫用。"""

    exam_id: str = Field("", description="Exam 表主鍵 exam_id（字串）")
    exam_tab_id: str = Field("", description="create-exam 回傳的 exam_tab_id；與 exam_id 二擇一")
    exam_quiz_id: str = Field("", description="選填，寫入 Exam_Answer 表 exam_quiz_id")
    quiz_content: str = Field(..., description="測驗題目內容（與 Exam_Quiz 表 quiz_content 一致）")
    answer: str = Field(..., description="學生回答")
    llm_api_key: str = Field(..., description="LLM API Key")


# 非同步評分結果暫存：job_id -> {"status": "pending"|"ready"|"error", "result": dict|None, "error": str|None}
_exam_grade_job_results: dict[str, dict[str, Any]] = {}


def _exam_grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    student_answer: str,
    *,
    exam_id: int = 0,
    exam_tab_id: str = "",
    person_id: str = "",
    exam_quiz_id: int = 0,
) -> None:
    """背景執行評分，結果寫入 _exam_grade_job_results[job_id]，並寫入 public.Exam_Answer 表。"""
    qtype = "short_answer"
    try:
        result = _run_grade_job(work_dir, api_key, quiz_content, student_answer, qtype)
        result_dict = result.model_dump()
        answer_id_val = None
        try:
            supabase = get_supabase()
            answer_feedback_json = json.dumps(result_dict, ensure_ascii=False)
            answer_row = {
                "exam_id": exam_id,
                "exam_tab_id": exam_tab_id or "",
                "exam_quiz_id": exam_quiz_id,
                "person_id": person_id or "",
                "student_answer": student_answer or "",
                "answer_grade": result.score,
                "answer_feedback_metadata": answer_feedback_json,
                "answer_metadata": result_dict,
            }
            ins = supabase.table("Exam_Answer").insert(answer_row).execute()
            if ins.data and len(ins.data) > 0:
                answer_id_val = ins.data[0].get("exam_answer_id")
        except Exception:
            pass
        if answer_id_val is not None:
            result_dict["exam_answer_id"] = answer_id_val
        _exam_grade_job_results[job_id] = {
            "status": "ready",
            "result": result_dict,
            "error": None,
        }
    except Exception as e:
        _exam_grade_job_results[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e),
        }
    finally:
        _cleanup_grade_workspace(work_dir)


@router.post("/create-exam")
def create_exam(body: CreateExamRequest):
    """
    建立一筆 Exam 資料。exam_tab_id 可選，未傳則由後端產生。
    回傳 exam_id、exam_tab_id、person_id、exam_name、created_at。
    """
    fid = (body.exam_tab_id or "").strip()
    if not fid:
        fid = generate_tab_id(body.person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")

    person_id = (body.person_id or "").strip()
    exam_name = (body.exam_name or "").strip()

    supabase = get_supabase()
    insert_row = {
        "exam_tab_id": fid,
        "person_id": person_id,
        "exam_name": exam_name,
        "deleted": False,
    }
    ins = supabase.table("Exam").insert(insert_row).execute()
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="建立 Exam 失敗")

    row = ins.data[0]
    return {
        "exam_id": row.get("exam_id"),
        "exam_tab_id": row.get("exam_tab_id", fid),
        "person_id": row.get("person_id", person_id),
        "exam_name": row.get("exam_name", exam_name),
        "created_at": row.get("created_at"),
    }


@router.post("/delete/{exam_tab_id}", status_code=200)
def delete_exam(
    exam_tab_id: str = PathParam(..., description="要刪除的 Exam 的 exam_tab_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    POST /exam/delete/{exam_tab_id}，person_id 請帶 Header X-Person-Id。
    軟刪除：將 Exam 表該筆 deleted 設為 true。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (exam_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")
    supabase = get_supabase()
    r = supabase.table("Exam").select("exam_id").eq("exam_tab_id", fid).eq("person_id", pid).eq("deleted", False).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="找不到該 exam_tab_id 的 Exam 資料")
    supabase.table("Exam").update({"deleted": True, "updated_at": _now_utc_iso()}).eq("exam_tab_id", fid).eq("person_id", pid).execute()
    return {
        "message": "已將 Exam 標記為刪除",
        "exam_tab_id": fid,
        "person_id": pid,
    }


def _exam_rag_stem_from_rag_id(supabase, rag_id: int) -> tuple[str, str]:
    """由 rag_id 查 Rag 表，回傳 (stem, rag_zip_tab_id)。stem 取自 rag_metadata.outputs 第一筆的 rag_tab_id。"""
    rag_rows = supabase.table("Rag").select("rag_metadata").eq("rag_id", rag_id).eq("deleted", False).execute()
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    meta = rag_rows.data[0].get("rag_metadata")
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")
    stem = (outputs[0].get("rag_tab_id") or "").strip() if isinstance(outputs[0], dict) else ""
    if not stem:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少 rag_tab_id")
    return stem, f"{stem}_rag"


@router.post("/generate-quiz")
def exam_generate_quiz(body: ExamGenerateQuizRequest):
    """
    傳入 llm_api_key、exam_id 或 exam_tab_id（二擇一）。程式依該 Exam 的 person_id 取得 for_exam=true 的 Rag，依其 rag_metadata.outputs 查找 RAG ZIP 出題。
    出題成功後寫入 public.Exam_Quiz 表；回傳 JSON 含 quiz_content, quiz_hint, reference_answer、exam_quiz_id 等。
    """
    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 llm_api_key")

    supabase = get_supabase()
    exam_id = body.exam_id or 0
    raw_tab = body.exam_tab_id
    exam_tab_id = (raw_tab or "").strip() if isinstance(raw_tab, str) else (str(raw_tab).strip() if raw_tab else "")
    if exam_id <= 0 and (not exam_tab_id or exam_tab_id == "0"):
        raise HTTPException(status_code=400, detail="請傳入 exam_id 或 exam_tab_id")

    if exam_id > 0:
        exam_rows = supabase.table("Exam").select("exam_id, exam_tab_id, person_id").eq("exam_id", exam_id).eq("deleted", False).execute()
    else:
        exam_rows = supabase.table("Exam").select("exam_id, exam_tab_id, person_id").eq("exam_tab_id", exam_tab_id).eq("deleted", False).execute()
    if not exam_rows.data or len(exam_rows.data) == 0:
        raise HTTPException(status_code=404, detail="找不到對應的 Exam 資料")
    row = exam_rows.data[0]
    exam_id = int(row.get("exam_id") or 0)
    exam_tab_id = (row.get("exam_tab_id") or "").strip()
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該 Exam 的 person_id 為空")

    # 取得該使用者的 for_exam Rag
    rag_rows = supabase.table("Rag").select("rag_id, system_prompt_instruction").eq("person_id", person_id).eq("for_exam", True).eq("deleted", False).execute()
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail="找不到該使用者的 for_exam Rag，請先設定供測驗使用的 RAG")
    rag_id = int(rag_rows.data[0].get("rag_id") or 0)
    system_prompt_instruction = (rag_rows.data[0].get("system_prompt_instruction") or "").strip()
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="該筆 for_exam Rag 的 system_prompt_instruction 未設定，請在 build-rag-zip 傳入")

    stem, rag_zip_tab_id = _exam_rag_stem_from_rag_id(supabase, rag_id)
    path = get_zip_path(rag_zip_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（tab_id={rag_zip_tab_id}）")

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
            "rag_name": stem,
            "filename": f"{stem}.zip",
        }

        quiz_row: dict[str, Any] = {
            "exam_id": exam_id,
            "exam_tab_id": exam_tab_id,
            "person_id": person_id,
            "rag_id": rag_id,
            "quiz_level": body.quiz_level,
            "quiz_content": result.get("quiz_content") or "",
            "quiz_hint": result.get("quiz_hint") or "",
            "reference_answer": result.get("reference_answer") or "",
        }
        quiz_row["quiz_metadata"] = result
        try:
            quiz_resp = supabase.table("Exam_Quiz").insert(quiz_row).execute()
            if quiz_resp.data and len(quiz_resp.data) > 0:
                result["exam_quiz_id"] = quiz_resp.data[0].get("exam_quiz_id")
                supabase.table("Exam_Quiz").update({"quiz_metadata": result}).eq("exam_quiz_id", result["exam_quiz_id"]).eq("exam_tab_id", exam_tab_id).execute()
        except Exception:
            pass
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz-grade")
async def exam_grade_submission(background_tasks: BackgroundTasks, body: ExamQuizGradeRequest):
    """
    傳入 llm_api_key、exam_id 或 exam_tab_id、exam_quiz_id、quiz_content、answer。
    程式依 Exam 取得 person_id，再依該使用者的 for_exam Rag 查找 RAG ZIP 評分。回傳 202 與 job_id；背景寫入 public.Exam_Answer。輪詢 GET /exam/quiz-grade-result/{job_id}。
    """
    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "請傳入 llm_api_key"})
    exam_id_str = (body.exam_id or "").strip()
    exam_tab_id = (body.exam_tab_id or "").strip()
    if not exam_id_str and not exam_tab_id:
        return JSONResponse(status_code=400, content={"error": "請傳入 exam_id 或 exam_tab_id"})

    supabase = get_supabase()
    if exam_id_str:
        try:
            exam_id_int = int(exam_id_str)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "exam_id 須為數字字串"})
        resp = supabase.table("Exam").select("exam_id, exam_tab_id, person_id").eq("exam_id", exam_id_int).eq("deleted", False).execute()
    else:
        resp = supabase.table("Exam").select("exam_id, exam_tab_id, person_id").eq("exam_tab_id", exam_tab_id).eq("deleted", False).execute()
    row = (resp.data or [None])[0] if resp.data else None
    if not row:
        return JSONResponse(status_code=404, content={"error": "找不到對應的 Exam 資料"})
    person_id = (row.get("person_id") or "").strip()
    exam_id = int(row.get("exam_id") or 0)
    exam_tab_id = (row.get("exam_tab_id") or "").strip()

    rag_rows = supabase.table("Rag").select("rag_id, rag_metadata").eq("person_id", person_id).eq("for_exam", True).eq("deleted", False).execute()
    if not rag_rows.data or len(rag_rows.data) == 0:
        return JSONResponse(status_code=404, content={"error": "找不到該使用者的 for_exam Rag"})
    meta = rag_rows.data[0].get("rag_metadata")
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        return JSONResponse(status_code=400, content={"error": "該筆 for_exam Rag 的 rag_metadata.outputs 為空"})
    stem = (outputs[0].get("rag_tab_id") or "").strip() if isinstance(outputs[0], dict) else ""
    if not stem:
        return JSONResponse(status_code=400, content={"error": "該筆 for_exam Rag 的 outputs 第一筆缺少 rag_tab_id"})
    rag_zip_tab_id = f"{stem}_rag"
    rag_zip_path = get_zip_path(rag_zip_tab_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP（tab_id={rag_zip_tab_id}）"})

    work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_exam_grade_"))
    zip_source_path = work_dir / "ref.zip"
    try:
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})

    try:
        exam_quiz_id_int = int((body.exam_quiz_id or "").strip()) if (body.exam_quiz_id or "").strip() else 0
    except ValueError:
        exam_quiz_id_int = 0
    job_id = str(uuid.uuid4())
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    quiz_content = body.quiz_content or ""
    answer = body.answer or ""
    background_tasks.add_task(
        _exam_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        answer,
        exam_id=exam_id,
        exam_tab_id=exam_tab_id,
        person_id=person_id,
        exam_quiz_id=exam_quiz_id_int,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str):
    """
    輪詢 Exam 評分結果。回傳 status: pending | ready | error；ready 時 result 含 exam_answer_id（對應 public.Exam_Answer 表），error 時 error 為錯誤訊息。
    """
    if job_id not in _exam_grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _exam_grade_job_results[job_id]
    return {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
    }
