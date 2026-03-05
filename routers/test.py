"""
Test API：對應 public.Test / Quiz / Answer 表。
- GET /test/tests：列出 Test 表（格式同 GET /rag/rags），每筆含 quizzes（每題帶 answers）與頂層 answers。
- POST /test/create-test：建立一筆 Test 資料。
- POST /test/generate-quiz：依 test_tab_id 與 rag_name 查找 RAG ZIP 出題，寫入 Quiz（rag_id=test_id）。
- POST /test/quiz-grade：非同步評分，寫入 Answer；輪詢 GET /test/quiz-grade-result/{job_id}。
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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from utils.storage import generate_tab_id, get_zip_path
from utils.supabase_client import get_supabase

from routers.grade import _run_grade_job, _cleanup_grade_workspace

router = APIRouter(prefix="/test", tags=["test"])


# --- GET /test/tests（格式同 /rag/rags）---

def _tests_table_select(exclude_deleted: bool = True) -> list[dict]:
    """查詢 Test 表，exclude_deleted=True 時僅回傳 deleted=False。"""
    supabase = get_supabase()
    q = supabase.table("Test").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    resp = q.execute()
    return resp.data or []


def _quizzes_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id（此處為 test_id）查詢 Quiz 表，回傳 rag_id -> list of quiz。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Quiz").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            try:
                out.setdefault(int(rid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


def _answers_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id（此處為 test_id）查詢 Answer 表，回傳 rag_id -> list of answer。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Answer").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            try:
                out.setdefault(int(rid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


class ListTestResponse(BaseModel):
    """GET /test/tests 回應：Test 表全部資料，每筆另含關聯的 Quiz（quizzes，每題帶 answers）與頂層 Answer（answers）。"""
    tests: list[dict]
    count: int


@router.get("/tests", response_model=ListTestResponse)
def list_tests(
    person_id: Optional[str] = Query(None, description="選填，篩選 person_id；未傳則回傳全部"),
):
    """
    列出 Test 表內容，僅回傳 deleted=False 的資料；每筆 Test 含表上所有欄位，並帶關聯的 Quiz（quizzes，每題帶 answers）與頂層 answers。
    格式同 GET /rag/rags。
    """
    try:
        data = _tests_table_select(exclude_deleted=True)
        if person_id is not None and str(person_id).strip():
            pid = str(person_id).strip()
            data = [r for r in data if (r.get("person_id") or "").strip() == pid]
        test_ids = []
        for row in data:
            tid = row.get("test_id")
            if tid is not None:
                try:
                    test_ids.append(int(tid))
                except (TypeError, ValueError):
                    pass
        test_ids = list(dict.fromkeys(test_ids))
        quizzes_by_test = _quizzes_by_rag_id(test_ids)
        answers_by_test = _answers_by_rag_id(test_ids)
        # 依 quiz_id 彙總 answers，供每筆 quiz 帶關聯的 answers
        answers_by_quiz_id: dict[int, list[dict]] = {}
        for tid in test_ids:
            for a in answers_by_test.get(tid, []):
                qid = a.get("quiz_id")
                if qid is not None:
                    try:
                        qid_int = int(qid)
                        answers_by_quiz_id.setdefault(qid_int, []).append(a)
                    except (TypeError, ValueError):
                        pass
        for row in data:
            tid = row.get("test_id")
            tid_int = int(tid) if tid is not None else None
            row_quizzes = quizzes_by_test.get(tid_int, []) if tid_int is not None else []
            for quiz in row_quizzes:
                qid = quiz.get("quiz_id")
                qid_int = int(qid) if qid is not None else None
                quiz["answers"] = answers_by_quiz_id.get(qid_int, []) if qid_int is not None else []
            row["quizzes"] = row_quizzes
            row["answers"] = answers_by_test.get(tid_int, []) if tid_int is not None else []
        return ListTestResponse(tests=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateTestRequest(BaseModel):
    """POST /test/create-test 請求 body。"""

    test_tab_id: str | None = Field(None, description="選填；未傳則由後端產生（格式同 tab_id）")
    person_id: str = Field("", description="選填，寫入 Test 表 person_id")
    test_name: str = Field("", description="測驗名稱，寫入 Test 表 test_name")


class TestGenerateQuizRequest(BaseModel):
    """POST /test/generate-quiz 請求 body。"""

    llm_api_key: str = Field(..., description="OpenAI API Key，由呼叫端傳入")
    test_tab_id: str = Field(..., description="create-test 回傳的 test_tab_id（Test 表識別）")
    rag_name: str = Field(..., description="如 220222_220301；程式會以 {rag_name}_rag 查找 RAG ZIP")
    system_prompt_instruction: str = Field(..., description="出題系統指令（Test 表無此欄，必填）")
    course_name: str = Field(..., description="課程名稱")
    quiz_level: int = Field(0, description="難度等級，寫入 Quiz 表 quiz_level")
    quiz_type: int = Field(0, description="題型代碼，寫入 Quiz 表 quiz_type")


class TestQuizGradeRequest(BaseModel):
    """POST /test/quiz-grade 請求 body。"""

    llm_api_key: str = Field(..., description="OpenAI API Key，由呼叫端傳入")
    test_tab_id: str = Field(..., description="create-test 回傳的 test_tab_id")
    rag_name: str = Field(..., description="如 220222_220301；程式會以 {rag_name}_rag 查找 RAG ZIP")
    quiz_content: str = Field(..., description="測驗題目內容（與 Quiz 表 quiz_content 一致）")
    student_answer: str = Field(..., description="學生回答")
    qtype: str = Field(..., description="題型，如 short_answer")
    course_name: str = Field("", description="選填，寫入 Answer 表 course_name")
    quiz_id: int = Field(0, description="選填，寫入 Answer 表 quiz_id；若已知題目對應的 quiz_id 可傳入")


# 非同步評分結果暫存：job_id -> {"status": "pending"|"ready"|"error", "result": dict|None, "error": str|None}
_test_grade_job_results: dict[str, dict[str, Any]] = {}


def _test_grade_job_background(
    job_id: str,
    work_dir: Path,
    api_key: str,
    quiz_content: str,
    student_answer: str,
    qtype: str,
    *,
    test_id: int = 0,
    test_tab_id: str = "",
    rag_name: str = "",
    person_id: str = "",
    course_name: str = "",
    quiz_id: int = 0,
) -> None:
    """背景執行評分，結果寫入 _test_grade_job_results[job_id]，並寫入 public.Answer 表。"""
    try:
        result = _run_grade_job(work_dir, api_key, quiz_content, student_answer, qtype)
        result_dict = result.model_dump()
        answer_id_val = None
        try:
            supabase = get_supabase()
            answer_feedback_json = json.dumps(result_dict, ensure_ascii=False)
            answer_row = {
                "quiz_id": quiz_id,
                "rag_id": test_id,
                "tab_id": test_tab_id or "",
                "person_id": person_id or "",
                "course_name": course_name or "",
                "rag_name": rag_name or "",
                "student_answer": student_answer or "",
                "answer_grade": result.score,
                "answer_feedback_metadata": answer_feedback_json,
                "answer_metadata": result_dict,
            }
            ins = supabase.table("Answer").insert(answer_row).execute()
            if ins.data and len(ins.data) > 0:
                answer_id_val = ins.data[0].get("answer_id")
        except Exception:
            pass
        if answer_id_val is not None:
            result_dict["answer_id"] = answer_id_val
        _test_grade_job_results[job_id] = {
            "status": "ready",
            "result": result_dict,
            "error": None,
        }
    except Exception as e:
        _test_grade_job_results[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e),
        }
    finally:
        _cleanup_grade_workspace(work_dir)


@router.post("/create-test")
def create_test(body: CreateTestRequest):
    """
    建立一筆 Test 資料。test_tab_id 可選，未傳則由後端產生。
    回傳 test_id、test_tab_id、person_id、test_name、created_at。
    """
    fid = (body.test_tab_id or "").strip()
    if not fid:
        fid = generate_tab_id(body.person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 test_tab_id")

    person_id = (body.person_id or "").strip()
    test_name = (body.test_name or "").strip()

    supabase = get_supabase()
    insert_row = {
        "test_tab_id": fid,
        "person_id": person_id,
        "test_name": test_name,
        "deleted": False,
    }
    ins = supabase.table("Test").insert(insert_row).execute()
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="建立 Test 失敗")

    row = ins.data[0]
    return {
        "test_id": row.get("test_id"),
        "test_tab_id": row.get("test_tab_id", fid),
        "person_id": row.get("person_id", person_id),
        "test_name": row.get("test_name", test_name),
        "created_at": row.get("created_at"),
    }


@router.post("/generate-quiz")
def test_generate_quiz(body: TestGenerateQuizRequest):
    """
    傳入 test_tab_id（create-test 的 test_tab_id）與 rag_name，程式以 {rag_name}_rag 查找 RAG ZIP 出題。
    llm_api_key、system_prompt_instruction 由請求 body 傳入。出題成功後寫入 public.Quiz 表（rag_id=test_id, tab_id=test_tab_id）。
    回傳 JSON 含 quiz_content, quiz_hint, reference_answer、quiz_id（若寫入成功）等。
    """
    test_tab_id = (body.test_tab_id or "").strip()
    if not test_tab_id:
        raise HTTPException(status_code=400, detail="請傳入 test_tab_id")

    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 llm_api_key")

    supabase = get_supabase()
    test_rows = supabase.table("Test").select("test_id, person_id").eq("test_tab_id", test_tab_id).eq("deleted", False).execute()
    if not test_rows.data or len(test_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 test_tab_id={test_tab_id} 的 Test 資料")

    row = test_rows.data[0]
    test_id = int(row.get("test_id") or 0)
    person_id = (row.get("person_id") or "").strip()

    rag_name = (body.rag_name or "").strip()
    if not rag_name:
        raise HTTPException(status_code=400, detail="請傳入 rag_name（如 220222_220301）")

    rag_zip_tab_id = f"{rag_name}_rag"
    path = get_zip_path(rag_zip_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_name={rag_name}（tab_id={rag_zip_tab_id}）")

    system_prompt_instruction = (body.system_prompt_instruction or "").strip()
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="請傳入 system_prompt_instruction（出題系統指令，必填）")

    course_name = (body.course_name or "").strip()
    if not course_name:
        raise HTTPException(status_code=400, detail="請傳入 course_name")

    try:
        from utils.quiz_gen import generate_quiz
        result = generate_quiz(
            path,
            api_key=api_key,
            quiz_level=body.quiz_level,
            system_prompt_instruction=system_prompt_instruction,
            course_name=course_name,
            quiz_type=body.quiz_type,
        )
        result["system_prompt_instruction"] = system_prompt_instruction
        result["quiz_level"] = body.quiz_level
        result["quiz_type"] = body.quiz_type
        result["rag_output"] = {
            "rag_name": rag_name,
            "filename": f"{rag_name}.zip",
        }

        quiz_row: dict[str, Any] = {
            "rag_id": test_id,
            "tab_id": test_tab_id,
            "person_id": person_id,
            "course_name": course_name,
            "system_prompt_instruction": system_prompt_instruction,
            "rag_name": rag_name,
            "quiz_level": body.quiz_level,
            "quiz_content": result.get("quiz_content") or "",
            "quiz_hint": result.get("quiz_hint") or "",
            "reference_answer": result.get("reference_answer") or "",
            "quiz_type": body.quiz_type,
        }
        quiz_row["quiz_metadata"] = result
        try:
            quiz_resp = supabase.table("Quiz").insert(quiz_row).execute()
            if quiz_resp.data and len(quiz_resp.data) > 0:
                result["quiz_id"] = quiz_resp.data[0].get("quiz_id")
                supabase.table("Quiz").update({"quiz_metadata": result}).eq("quiz_id", result["quiz_id"]).eq("tab_id", test_tab_id).execute()
        except Exception:
            pass
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz-grade")
async def test_grade_submission(background_tasks: BackgroundTasks, body: TestQuizGradeRequest):
    """
    傳入 test_tab_id 與 rag_name，程式以 {rag_name}_rag 查找 RAG ZIP 進行評分。
    驗證後回傳 202 與 job_id；實際評分在背景執行，完成後寫入 public.Answer 表。前端以 GET /test/quiz-grade-result/{job_id} 輪詢，ready 時 result 含 answer_id。
    """
    test_tab_id = (body.test_tab_id or "").strip()
    if not test_tab_id:
        return JSONResponse(status_code=400, content={"error": "請傳入 test_tab_id"})

    api_key = (body.llm_api_key or "").strip()
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "請傳入 llm_api_key"})

    supabase = get_supabase()
    resp = supabase.table("Test").select("test_id, person_id").eq("test_tab_id", test_tab_id).eq("deleted", False).execute()
    row = (resp.data or [None])[0] if resp.data else None
    if not row:
        return JSONResponse(status_code=404, content={"error": f"找不到 test_tab_id={test_tab_id} 的 Test 資料"})
    person_id = (row.get("person_id") or "").strip()
    test_id = int(row.get("test_id") or 0)

    rag_name = (body.rag_name or "").strip()
    if not rag_name:
        return JSONResponse(status_code=400, content={"error": "請傳入 rag_name（如 220222_220301）"})

    rag_zip_tab_id = f"{rag_name}_rag"
    rag_zip_path = get_zip_path(rag_zip_tab_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"找不到 RAG ZIP，請確認 rag_name={rag_name}（tab_id={rag_zip_tab_id}）"},
        )

    work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_test_grade_"))
    zip_source_path = work_dir / "ref.zip"
    try:
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})

    job_id = str(uuid.uuid4())
    _test_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    quiz_content = body.quiz_content or ""
    student_answer = body.student_answer or ""
    qtype = body.qtype or ""
    course_name = (body.course_name or "").strip()
    quiz_id = body.quiz_id or 0

    background_tasks.add_task(
        _test_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        student_answer,
        qtype,
        test_id=test_id,
        test_tab_id=test_tab_id,
        rag_name=rag_name,
        person_id=person_id,
        course_name=course_name,
        quiz_id=quiz_id,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["test"])
async def get_test_grade_result(job_id: str):
    """
    輪詢 Test 評分結果。回傳 status: pending | ready | error；ready 時 result 含 answer_id（對應 public.Answer 表），error 時 error 為錯誤訊息。
    """
    if job_id not in _test_grade_job_results:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "result": None,
                "error": "job not found（可能為服務重啟或冷啟動，請重新送出評分）",
            },
        )
    data = _test_grade_job_results[job_id]
    return {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
    }
