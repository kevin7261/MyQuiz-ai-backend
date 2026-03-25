"""
Exam API 模組。對應 public.Exam / Exam_Quiz / Exam_Answer 表。
- GET /exam/exams：列出 Exam 表（格式同 GET /rag/rags），query `local` 篩選 Exam.local，未傳時依連線是否本機判定；每筆含 quizzes（每題帶 answers）與頂層 answers。
- POST /exam/create-exam：建立一筆 Exam 資料（可傳 local，用法同 POST /rag/create-unit）。
- POST /exam/create-quiz：依 exam_tab_id 與 rag_id 查找 RAG ZIP 出題，寫入 Exam_Quiz。
- POST /exam/grade-quiz 或 POST /exam/quiz-grade（同義）：非同步評分並寫入 Exam_Answer（與 /rag 評分流程一致；寫入失敗時輪詢 status 為 error）；輪詢 GET /exam/quiz-grade-result/{job_id}。
- POST /exam/delete/{exam_tab_id}：軟刪除該筆 Exam（deleted=true）。
"""

# 引入 json 用於序列化回傳
import json
# 引入 logging 用於列出 Exam 錯誤紀錄（與 GET /rag/rags 一致）
import logging
# 引入 shutil 用於複製檔案
import shutil
# 引入 tempfile 用於暫存目錄
import tempfile
# 引入 uuid 用於產生 job_id
import uuid
# 引入 zipfile 用於驗證 ZIP
import zipfile
# 引入 Path 用於路徑
from pathlib import Path
# 引入 Any、Optional 型別
from typing import Any, Optional

# 引入 FastAPI 相關
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Path as PathParam, Query, Request
# 引入 JSONResponse、Response
from fastapi.responses import JSONResponse, Response
# 引入 Pydantic 的 BaseModel、Field
from pydantic import AliasChoices, BaseModel, Field, field_validator

# UTC 時間
from utils.datetime_utils import now_utc_iso
# 轉成可 JSON 序列化（與 GET /rag/rags 一致）
from utils.json_utils import to_json_safe
# 系統 LLM API Key（Exam 使用系統設定，非個人）
from utils.llm_api_key_utils import get_llm_api_key
# 由 rag_id 取得 stem、rag_zip_tab_id
from utils.rag_common import get_rag_stem_from_rag_id
# 供測驗 RAG：System_Setting rag_id、本機判定（與 Rag.local 篩選一致）
from utils.rag_exam_setting import fetch_exam_rag_id_from_settings, is_localhost_request
# 儲存：generate_tab_id、get_zip_path
from utils.storage import generate_tab_id, get_zip_path
# Supabase 客戶端
from utils.supabase_client import get_supabase

# 從 grade 模組引入共用評分邏輯
from routers.grade import _run_grade_job_background, _insert_exam_answer, _cleanup_grade_workspace

# 建立路由，前綴 /exam
router = APIRouter(prefix="/exam", tags=["exam"])


# --- GET /exam/exams（格式同 /rag/rags）---


def _exam_default_row(
    exam_tab_id: str,
    *,
    exam_name: str = "",
    person_id: str = "",
    local: bool = False,
) -> dict[str, Any]:
    """Exam 表新增一筆時的預設欄位；鍵順序同 public.Exam（不含 exam_id/created_at）。"""
    return {
        "exam_tab_id": exam_tab_id,
        "exam_name": exam_name,
        "person_id": person_id,
        "local": local,
        "deleted": False,
        "updated_at": now_utc_iso(),
    }


def _exams_table_select(
    exclude_deleted: bool = True,
    *,
    local_match: bool | None = None,
) -> list[dict]:
    """查詢 Exam 表全部列。exclude_deleted=True 時僅回傳 deleted=False。local_match 若指定則僅回傳 Exam.local 與其相符的列。"""
    supabase = get_supabase()
    q = supabase.table("Exam").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    resp = q.execute()
    return resp.data or []


def _exams_by_ids(exam_ids: list[int]) -> list[dict]:
    """依 exam_id 查詢 Exam 表，回傳對應的 Exam 列（僅 deleted=False）。"""
    if not exam_ids:
        return []
    supabase = get_supabase()
    resp = supabase.table("Exam").select("*").in_("exam_id", exam_ids).eq("deleted", False).execute()
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


def _all_exam_quizzes() -> list[dict]:
    """查詢 Exam_Quiz 表全部筆數，回傳所有題目（供 course analysis 使用）。"""
    supabase = get_supabase()
    resp = supabase.table("Exam_Quiz").select("*").execute()
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
    """GET /exam/exams 回應：Exam 表全部資料，每筆另含關聯的 Exam_Quiz（quizzes，每題帶一筆 answer）與頂層 Exam_Answer（answers）。"""
    exams: list[dict]
    count: int


@router.get("/exams", response_model=ListExamResponse)
def list_exams(
    request: Request,
    person_id: Optional[str] = Query(None, description="選填，篩選 person_id；未傳則回傳全部"),
    local: bool | None = Query(
        None,
        description="僅回傳 Exam.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false（與 GET /rag/rags 一致）",
    ),
):
    """
    列出 Exam 表內容，僅回傳 deleted=False 的資料，且 Exam.local 須與 query `local` 相符（未傳 `local` 時依連線是否本機自動判定）。
    每筆 Exam 含表上所有欄位，並帶關聯的 Exam_Quiz（quizzes，每題帶一筆 answer）與頂層 answers。
    格式同 GET /rag/rags。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _exams_table_select(exclude_deleted=True, local_match=local_filter)
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
                # 每題 quiz 只帶一筆 answer（取第一筆）
                raw_answers = (answers_by_quiz_id.get(qid_int, []) or []) if qid_int is not None else []
                quiz["answers"] = raw_answers[:1]
            row["quizzes"] = row_quizzes
            row["answers"] = answers_by_exam.get(eid_int, []) if eid_int is not None else []
        data = to_json_safe(data)
        return ListExamResponse(exams=data, count=len(data))
    except Exception as e:
        logging.exception("GET /exam/exams 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Exam 失敗: {e!s}")


class CreateExamRequest(BaseModel):
    """POST /exam/create-exam：欄位順序同 public.Exam（exam_tab_id, exam_name, person_id, local；exam_id／created_at 由資料庫產生；insert 另帶 deleted, updated_at）。"""

    exam_tab_id: str | None = Field(None, description="選填；未傳則由後端產生（格式同 tab_id）")
    exam_name: str = Field("", description="測驗名稱，寫入 Exam 表 exam_name")
    person_id: str = Field("", description="選填，寫入 Exam 表 person_id")
    local: bool = Field(False, description="是否為本機 Exam，寫入 Exam 表 local 欄位")


class ExamGenerateQuizRequest(BaseModel):
    """POST /exam/create-quiz；欄位順序對齊 public.Exam_Quiz 中由客戶端提供的子集：exam_id, exam_tab_id, person_id（後端自 Exam 帶入）, rag_id（後端帶入）, unit_name, …"""

    exam_id: int = Field(0, description="Exam 表主鍵 exam_id")
    exam_tab_id: str = Field("", description="create-exam 回傳的 exam_tab_id（varchar）；與 exam_id 二擇一")
    unit_name: str = Field(
        "",
        description="選填；指定供測驗 Rag 的 rag_metadata.outputs 中某一上傳單元（與 POST /rag/build-rag-zip 的 outputs[].unit_name 一致）。未傳或空字串則使用第一筆輸出",
    )
    quiz_level: str = Field("", description="難度／層級（字串），用於出題提示並寫入 quiz_metadata")

    @field_validator("quiz_level", mode="before")
    @classmethod
    def _quiz_level_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/grade-quiz：寫入 public.Exam_Answer 時對應 exam_id, exam_tab_id, exam_quiz_id, person_id（後端自 Exam 帶入）, quiz_answer；評分後寫入 quiz_grade、quiz_grade_metadata。LLM API Key 由系統設定取得。"""

    exam_id: str = Field("", description="Exam 表主鍵 exam_id（字串）")
    exam_tab_id: str = Field("", description="create-exam 回傳的 exam_tab_id（varchar）；與 exam_id 二擇一")
    exam_quiz_id: str = Field("", description="選填，寫入 Exam_Answer.exam_quiz_id")
    quiz_content: str = Field(..., description="測驗題目內容（與 Exam_Quiz.quiz_content 一致）")
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Exam_Answer.quiz_answer）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


# 非同步評分結果暫存：job_id -> {"status": "pending"|"ready"|"error", "result": dict|None, "error": str|None}
_exam_grade_job_results: dict[str, dict[str, Any]] = {}


@router.post("/create-exam")
def create_exam(body: CreateExamRequest):
    """
    建立一筆 Exam 資料。exam_tab_id 可選，未傳則由後端產生；local 選填，預設 false（與 create-unit 一致）。
    回傳 exam_id、exam_tab_id、person_id、exam_name、local、created_at。
    """
    fid = (body.exam_tab_id or "").strip()
    if not fid:
        fid = generate_tab_id(body.person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")

    person_id = (body.person_id or "").strip()
    exam_name = (body.exam_name or "").strip()

    supabase = get_supabase()
    ins = (
        supabase.table("Exam")
        .insert(
            _exam_default_row(
                fid,
                exam_name=exam_name,
                person_id=person_id,
                local=body.local,
            )
        )
        .execute()
    )
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="建立 Exam 失敗")

    row = ins.data[0]
    return {
        "exam_id": row.get("exam_id"),
        "exam_tab_id": row.get("exam_tab_id", fid),
        "exam_name": row.get("exam_name", exam_name),
        "person_id": row.get("person_id", person_id),
        "local": row.get("local", body.local),
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
    supabase.table("Exam").update({"deleted": True, "updated_at": now_utc_iso()}).eq("exam_tab_id", fid).eq("person_id", pid).execute()
    return {
        "message": "已將 Exam 標記為刪除",
        "exam_tab_id": fid,
        "person_id": pid,
    }


@router.post("/create-quiz", summary="Exam Create Quiz", operation_id="exam_create_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def exam_create_quiz(request: Request, body: ExamGenerateQuizRequest):
    """
    傳入 exam_id 或 exam_tab_id（二擇一）、quiz_level；可傳 unit_name 指定 outputs 中哪一個上傳單元（與 build-rag-zip 的 outputs[].unit_name 一致），未傳則用第一筆。
    LLM API Key 由系統設定（/system-settings/llm-api-key）取得；請先於系統設定填寫。
    依連線讀取 System_Setting（rag_localhost / rag_deploy）的 rag_id，取得對應 Rag，依選定單元載入 RAG ZIP 出題。
    出題成功後寫入 public.Exam_Quiz 表；回傳 JSON 含 quiz_content, quiz_hint, quiz_reference_answer、exam_quiz_id 等。
    """
    supabase = get_supabase()
    exam_id = body.exam_id or 0
    exam_tab_id = (body.exam_tab_id or "").strip()
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
    api_key = get_llm_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請先於系統設定（/system-settings/llm-api-key）填寫 LLM API Key",
        )

    _, rag_id_from_setting = fetch_exam_rag_id_from_settings(supabase, request)
    if rag_id_from_setting is None or rag_id_from_setting <= 0:
        raise HTTPException(
            status_code=404,
            detail="尚未設定供測驗用 RAG rag_id，請以 PUT /system-settings/rag-for-exam-localhost 或 rag-for-exam-deploy 寫入",
        )
    rag_rows = (
        supabase.table("Rag")
        .select("rag_id, system_prompt_instruction")
        .eq("rag_id", rag_id_from_setting)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id_from_setting} 的 Rag 資料，或已刪除")
    rag_id = int(rag_rows.data[0].get("rag_id") or 0)
    system_prompt_instruction = (rag_rows.data[0].get("system_prompt_instruction") or "").strip()
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="該筆供測驗 Rag 的 system_prompt_instruction 未設定，請在 build-rag-zip 傳入")

    unit_filter = (body.unit_name or "").strip() or None
    stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id, unit_name=unit_filter)
    # 取得 RAG ZIP 路徑（下載至暫存檔）
    path = get_zip_path(rag_zip_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（tab_id={rag_zip_tab_id}）")

    try:
        from utils.create_quiz import generate_quiz
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
            "unit_name": stem,
            "filename": f"{stem}.zip",
        }

        file_name = f"{stem}.zip"
        # 鍵順序同 public.Exam_Quiz（不含 exam_quiz_id / 時間戳）
        quiz_row: dict[str, Any] = {
            "exam_id": exam_id,
            "exam_tab_id": exam_tab_id,
            "person_id": person_id,
            "rag_id": rag_id,
            "unit_name": stem,
            "file_name": file_name,
            "quiz_content": result.get("quiz_content") or "",
            "quiz_hint": result.get("quiz_hint") or "",
            "quiz_answer_reference": result.get("quiz_reference_answer") or "",
            "quiz_metadata": result,
        }
        try:
            quiz_resp = supabase.table("Exam_Quiz").insert(quiz_row).execute()
            if quiz_resp.data and len(quiz_resp.data) > 0:
                result["exam_quiz_id"] = quiz_resp.data[0].get("exam_quiz_id")
                supabase.table("Exam_Quiz").update({"quiz_metadata": result}).eq("exam_quiz_id", result["exam_quiz_id"]).eq("exam_id", exam_id).eq("exam_tab_id", exam_tab_id).execute()
        except Exception:
            pass  # 與 POST /rag/create-quiz 相同：寫入題庫失敗仍回傳出題 JSON
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理從 Supabase Storage 下載的暫存檔
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/grade-quiz", summary="Exam Grade Quiz")
@router.post("/quiz-grade", summary="Exam Grade Quiz (alias: /quiz-grade)")
async def exam_grade_submission(request: Request, background_tasks: BackgroundTasks, body: ExamQuizGradeRequest):
    """
    傳入 exam_id 或 exam_tab_id、exam_quiz_id、quiz_content、quiz_answer。
    LLM API Key 由系統設定（/system-settings/llm-api-key）取得；請先於系統設定填寫。
    依連線讀取 System_Setting（rag_localhost / rag_deploy）的 rag_id；若帶 exam_quiz_id 則依該題 Exam_Quiz.unit_name 載入對應 RAG ZIP（與 create-quiz 指定 unit_name 一致），否則使用第一筆 outputs。
    回傳 202 與 job_id；背景寫入 public.Exam_Answer（與 POST /rag/grade-quiz 相同管線；寫入失敗則輪詢為 error）。輪詢 GET /exam/quiz-grade-result/{job_id}，ready 時 result 含 quiz_grade、quiz_comments 及 exam_answer_id。
    """
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
    api_key = get_llm_api_key()
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "請先於系統設定（/system-settings/llm-api-key）填寫 LLM API Key",
            },
        )

    _, rag_id_from_setting = fetch_exam_rag_id_from_settings(supabase, request)
    if rag_id_from_setting is None or rag_id_from_setting <= 0:
        return JSONResponse(
            status_code=404,
            content={
                "error": "尚未設定供測驗用 RAG rag_id，請以 PUT /system-settings/rag-for-exam-localhost 或 rag-for-exam-deploy 寫入",
            },
        )
    rag_rows = (
        supabase.table("Rag")
        .select("rag_id, rag_metadata")
        .eq("rag_id", rag_id_from_setting)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_rows.data or len(rag_rows.data) == 0:
        return JSONResponse(
            status_code=404,
            content={"error": f"找不到 rag_id={rag_id_from_setting} 的 Rag 資料，或已刪除"},
        )
    try:
        exam_quiz_id_int = int((body.exam_quiz_id or "").strip()) if (body.exam_quiz_id or "").strip() else 0
    except ValueError:
        exam_quiz_id_int = 0
    grade_unit_filter: str | None = None
    if exam_quiz_id_int > 0:
        qquiz = (
            supabase.table("Exam_Quiz")
            .select("unit_name")
            .eq("exam_quiz_id", exam_quiz_id_int)
            .eq("exam_id", exam_id)
            .eq("exam_tab_id", exam_tab_id)
            .limit(1)
            .execute()
        )
        if not qquiz.data:
            return JSONResponse(status_code=404, content={"error": f"找不到 exam_quiz_id={exam_quiz_id_int} 的 Exam_Quiz 資料"})
        grade_unit_filter = (qquiz.data[0].get("unit_name") or "").strip() or None
    try:
        rag_id = int(rag_rows.data[0].get("rag_id") or 0)
        stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id, unit_name=grade_unit_filter)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    # 取得 RAG ZIP 路徑（下載至暫存檔）
    rag_zip_path = get_zip_path(rag_zip_tab_id)
    if not rag_zip_path or not rag_zip_path.exists():
        return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP（tab_id={rag_zip_tab_id}）"})

    work_dir = Path(tempfile.mkdtemp(prefix="aiquiz_exam_grade_"))
    zip_source_path = work_dir / "ref.zip"
    extract_folder = work_dir / "extract"
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

    job_id = str(uuid.uuid4())
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    insert_fn = lambda rd, qa: _insert_exam_answer(rd, qa, exam_id=exam_id, exam_tab_id=exam_tab_id, person_id=person_id, exam_quiz_id=exam_quiz_id_int)
    background_tasks.add_task(
        _run_grade_job_background,
        job_id,
        work_dir,
        api_key,
        body.quiz_content or "",
        body.quiz_answer or "",
        _exam_grade_job_results,
        insert_fn,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.get("/quiz-grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str):
    """
    輪詢 Exam 評分結果（行為同 GET /rag/quiz-grade-result/{job_id}）。
    status: pending | ready | error；ready 時 result 含 quiz_grade、quiz_comments、exam_answer_id（已寫入 Exam_Answer）；
    error 時為 LLM／ZIP 例外，或 LLM 成功但寫入 Exam_Answer 失敗。
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
