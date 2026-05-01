"""
RAG 評分與出題 API 模組。
- POST /rag/tab/unit/quiz/llm-generate：依 rag_quiz_id 出題（LLM）；unit_type 1 僅 RAG ZIP 向量檢索；2/3/4 以 transcription 純生成；其餘載 RAG ZIP 向量檢索。
- POST /rag/tab/unit/quiz/llm-grade：非同步 RAG+LLM 評分（body 以 rag_id 置頂；quiz_content 可空，自 Rag_Quiz 讀題幹）；回傳 202 + job_id，輪詢 GET /rag/tab/unit/quiz/grade-result/{job_id}。
- POST /rag/tab/unit/quiz/for-exam：更新 Rag_Quiz.for_exam（body `for_exam` 預設 true；false 取消測驗用）。
- GET /rag/tab/unit/quiz/grade-result/{job_id}：輪詢評分結果（ready 時含 rag_quiz 整列）。
- GET /rag/transcript/text、audio、youtube：自 Storage upload ZIP 讀取逐字稿。
- GET /rag/unit/audio-file：自 upload ZIP 依單元資料夾回傳原始音訊 bytes（供 `<audio src>`；與 transcript/audio 相同之 rag_tab_id、folder_name）。
"""

import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from utils.quiz_generation import generate_quiz, generate_quiz_transcription_only

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from postgrest.exceptions import APIError
from dependencies.person_id import PersonId
from fastapi.responses import JSONResponse, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_rag_quiz_with_grade,
)
from utils.datetime_utils import now_taipei_iso
from utils.json_utils import to_json_safe
from utils.llm_api_key_utils import get_llm_api_key_for_person
from utils.media_transcript import (
    audio_media_type_for_suffix,
    transcribe_audio_bytes_deepgram,
    youtube_transcript_api_user_message,
    youtube_transcript_plain_text,
)
from utils.rag_faiss_zip import process_zip_to_docs
from utils.rag_stem_utils import get_rag_stem_from_rag_id, instruction_from_rag_row
from utils.rag_transcript_from_upload_zip import (
    pick_audio_from_upload_zip,
    read_single_transcript_text_from_upload_zip,
    read_upload_zip_bytes,
    read_youtube_video_id_from_upload_zip,
)
from utils.supabase_client import get_supabase
from utils.zip_storage import get_zip_path
from youtube_transcript_api._errors import (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

router = APIRouter(prefix="/rag", tags=["rag"])

_logger = logging.getLogger(__name__)

_grade_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateQuizRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate；請求 body 僅含 rag_quiz_id、quiz_name、quiz_user_prompt_text。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )


class QuizGradeRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade 請求 body。
    rag_id 置頂：用於載入 RAG ZIP 與權限；quiz_content 可省略，沿用該 Rag_Quiz 資料列題幹。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
    rag_tab_id: str = Field("", description="選填；後端以 Rag.rag_tab_id 為準")
    quiz_content: str = Field(
        "",
        description="選填；非空時優先並可寫入 Rag_Quiz；空則自該 rag_quiz_id 之 Rag_Quiz.quiz_content 讀取（須於庫記憶中有題幹）",
    )
    answer_user_prompt_text: str = Field(
        "",
        description="作答／批改 user prompt（可空）；寫入 Rag_Quiz.answer_user_prompt_text 並供評分 prompt 參考",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Quiz.answer_content）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


class RagQuizForExamRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/for-exam：欄位順序對齊 Rag_Quiz（主鍵與關聯欄）。
    以 rag_quiz_id 更新 Rag_Quiz.for_exam；若一併傳入 rag_tab_id／rag_unit_id（>0），須與該列一致。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"rag_quiz_id": 1, "for_exam": True},
                {"rag_quiz_id": 1, "for_exam": False},
            ],
        },
    )

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_tab_id: str = Field("", description="選填；與資料列 rag_tab_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    for_exam: bool = Field(True, description="true：標記為測驗用；false：取消測驗用")


class RagTranscriptMarkdownResponse(BaseModel):
    """GET /rag/transcript/text、audio、youtube 共用回傳：markdown 僅為正文，無額外標題或 meta 區塊。"""

    markdown: str = Field(..., description="正文純文字或原檔內容（無 # Transcript 包裝）")


# ---------------------------------------------------------------------------
# 路由內輔助（僅限此模組）
# ---------------------------------------------------------------------------

def _require_rag_tab_owner(person_id: str, rag_tab_id: str) -> None:
    """確認 Rag 列存在且 person_id 一致。"""
    rid = (rag_tab_id or "").strip()
    if not rid or "/" in rid or "\\" in rid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    supabase = get_supabase()
    sel = (
        supabase.table("Rag")
        .select("rag_tab_id")
        .eq("rag_tab_id", rid)
        .eq("person_id", person_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_tab_id，或已刪除／不屬於此 person_id",
        )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate
# ---------------------------------------------------------------------------

@router.post("/tab/unit/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def rag_llm_generate_quiz(body: GenerateQuizRequest, caller_person_id: PersonId):
    """
    Body：rag_quiz_id、quiz_name、quiz_user_prompt_text（後兩者可空字串）；
    rag_tab_id／rag_unit_id 由後端依 rag_quiz_id 自資料庫帶入；quiz_user_prompt_text 空則自該列 Rag_Quiz 讀取。
    unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcription。
    unit_type 2／3／4 時不載入 RAG ZIP，改以固定 system（`SYSTEM_PROMPT_FAISS_QUIZ`）與含逐字稿 **課程內容** 之 user（`USER_PROMPT_TRANSCRIPTION_COURSE`）出題，與 RAG 路徑結構一致。
    出題成功後更新 public.Rag_Quiz（quiz_name、quiz_*；並清空 answer_*）。
    """
    supabase = get_supabase()

    q_sel = (
        supabase.table("Rag_Quiz")
        .select("rag_quiz_id, rag_tab_id, rag_unit_id, quiz_user_prompt_text")
        .eq("rag_quiz_id", body.rag_quiz_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not q_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={body.rag_quiz_id} 的 Rag_Quiz")
    q_row = q_sel.data[0]
    qup_body = (body.quiz_user_prompt_text or "").strip()
    qup_db = (q_row.get("quiz_user_prompt_text") or "").strip()
    qup_for_llm = qup_body or qup_db
    source_rag_unit_id = int(q_row.get("rag_unit_id") or 0)
    if source_rag_unit_id <= 0:
        raise HTTPException(status_code=400, detail="該 rag_quiz_id 對應的 rag_unit_id 無效")

    try:
        unit_sel = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id, rag_tab_id, unit_name, folder_combination, transcription, unit_type")
            .eq("rag_unit_id", source_rag_unit_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            unit_sel = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id, rag_tab_id, unit_name, transcription, unit_type")
                .eq("rag_unit_id", source_rag_unit_id)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
        else:
            raise
    if not unit_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_unit_id={source_rag_unit_id} 的 Rag_Unit")
    unit_row = unit_sel.data[0]
    unit_filter = (
        (unit_row.get("folder_combination") or unit_row.get("unit_name") or "").strip() or None
    )
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

    row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
        supabase,
        rag_id,
        include_row=True,
        unit_name=unit_filter,
        rag_unit_id=source_rag_unit_id,
    )
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該筆 Rag 的 person_id 為空，無法取得 LLM API Key")
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Rag 出題")
    api_key = get_llm_api_key_for_person(person_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="該使用者尚未填寫 LLM API Key：請至個人設定填寫，或本機在 .env 設定 LLM_API_KEY／OPENAI_API_KEY",
        )
    transcription_text = (unit_row.get("transcription") or "").strip()
    if not transcription_text:
        transcription_text = instruction_from_rag_row(row)

    try:
        unit_type_val = int(unit_row.get("unit_type") or 0)
    except (TypeError, ValueError):
        unit_type_val = 0

    if unit_type_val in (2, 3, 4) and not transcription_text:
        raise HTTPException(
            status_code=400,
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 或 Rag 設定 transcription，或經 POST /rag/tab/build-rag-zip 寫入 Rag_Unit.transcription",
        )

    path: Path | None = None
    try:
        if unit_type_val in (2, 3, 4):
            result = generate_quiz_transcription_only(
                api_key=api_key,
                transcription=transcription_text,
                quiz_user_prompt_text=qup_for_llm,
            )
        else:
            path = get_zip_path(rag_zip_tab_id)
            if not path or not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（rag_tab_id={rag_zip_tab_id}）",
                )
            result = generate_quiz(
                path,
                api_key=api_key,
                quiz_user_prompt_text=qup_for_llm,
            )
        result["transcription"] = "" if unit_type_val == 1 else transcription_text
        result["rag_output"] = {
            "rag_tab_id": stem,
            "unit_name": stem,
            "filename": f"{stem}.zip",
        }
        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_answer_reference") or "").strip()
        result["quiz_content"] = qc
        result["quiz_hint"] = qh
        result["quiz_answer_reference"] = qref
        result["rag_quiz_id"] = body.rag_quiz_id
        qup_stored = qup_body if qup_body else qup_db
        qts = now_taipei_iso()
        body_quiz_name = body.quiz_name.strip()
        quiz_name = body_quiz_name or ((stem or "").strip() or (unit_row.get("unit_name") or "").strip() or "")
        result["quiz_name"] = quiz_name
        quiz_update: dict[str, Any] = {
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": qup_stored,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_user_prompt_text": "",
            "answer_content": "",
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
                "Rag_Quiz llm-generate 讀回驗證失敗 rag_quiz_id=%s（quiz_content 不一致）",
                body.rag_quiz_id,
            )
            raise HTTPException(
                status_code=500,
                detail="寫入 Rag_Quiz 未生效（更新後讀回題幹與預期不符）。請檢查 RLS 政策或是否以 anon key 連線導致更新被擋。",
            )
        return Response(
            content=json.dumps(result, ensure_ascii=False).encode("utf-8"),
            media_type="application/json; charset=utf-8",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-grade
# ---------------------------------------------------------------------------

@router.post("/tab/unit/quiz/llm-grade", summary="Rag Grade Quiz")
async def grade_submission(background_tasks: BackgroundTasks, body: QuizGradeRequest, caller_person_id: PersonId):
    """
    非同步評分：Body 以 rag_id、rag_quiz_id 為核心；quiz_content 可省略（自 Rag_Quiz 讀）。
    unit_type 2／3／4 時以 transcription 純 LLM 批改；其餘依 rag_id 載入 RAG ZIP。
    回傳 202 + job_id；輪詢 GET /rag/tab/unit/quiz/grade-result/{job_id}。
    """
    rag_id_str = (body.rag_id or "").strip()
    if not rag_id_str:
        return JSONResponse(status_code=400, content={"error": "請傳入 rag_id"})
    try:
        rag_id_int = int(rag_id_str)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "rag_id 須為數字字串"})

    supabase = get_supabase()
    try:
        row, stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id_int, include_row=True)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        return JSONResponse(status_code=400, content={"error": "該筆 Rag 的 person_id 為空，無法取得 LLM API Key"})
    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Rag 評分"})

    try:
        rag_quiz_id_int = int((body.rag_quiz_id or "").strip()) if (body.rag_quiz_id or "").strip() else 0
    except ValueError:
        rag_quiz_id_int = 0
    if rag_quiz_id_int <= 0:
        return JSONResponse(status_code=400, content={"error": "rag_quiz_id 必填且須為大於 0 的整數（對應 Rag_Quiz 主鍵）"})

    api_key = get_llm_api_key_for_person(person_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "該使用者尚未填寫 LLM API Key：請至個人設定填寫，或本機在 .env 設定 LLM_API_KEY／OPENAI_API_KEY",
            },
        )

    rq_sel = (
        supabase.table("Rag_Quiz")
        .select("rag_unit_id, quiz_user_prompt_text, quiz_content")
        .eq("rag_quiz_id", rag_quiz_id_int)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rq_sel.data:
        return JSONResponse(status_code=404, content={"error": f"找不到 rag_quiz_id={rag_quiz_id_int} 的 Rag_Quiz"})
    rq_row = rq_sel.data[0]
    quiz_user_prompt_db = (rq_row.get("quiz_user_prompt_text") or "").strip()
    qc_from_body = (body.quiz_content or "").strip()
    qc_from_db = (rq_row.get("quiz_content") or "").strip()
    quiz_content_resolved = qc_from_body or qc_from_db
    if not quiz_content_resolved:
        return JSONResponse(
            status_code=400,
            content={
                "error": "缺少測驗題幹：請於請求傳入 quiz_content，或先於該 Rag_Quiz 設定 quiz_content。",
            },
        )
    grade_unit_type = 0
    transcription_text = ""
    try:
        ruid_raw = rq_row.get("rag_unit_id")
        ruid_i = int(ruid_raw) if ruid_raw is not None else 0
    except (TypeError, ValueError):
        ruid_i = 0
    if ruid_i > 0:
        uu = (
            supabase.table("Rag_Unit")
            .select("unit_type, transcription")
            .eq("rag_unit_id", ruid_i)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if uu.data:
            u0 = uu.data[0]
            try:
                grade_unit_type = int(u0.get("unit_type") or 0)
            except (TypeError, ValueError):
                grade_unit_type = 0
            transcription_text = (u0.get("transcription") or "").strip()
    if not transcription_text:
        transcription_text = instruction_from_rag_row(row)

    transcription_grade: str | None = None

    if grade_unit_type in (2, 3, 4):
        if not transcription_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcription 未設定：請於 Rag_Unit 或 Rag 設定 transcription（單元 2／3／4）"},
            )
        transcription_grade = transcription_text
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_grade_tx_"))
    else:
        rag_zip_path = get_zip_path(rag_zip_tab_id)
        if not rag_zip_path or not rag_zip_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"找不到 RAG ZIP，請確認 rag_id={rag_id_str}（tab_id={rag_zip_tab_id}）"},
            )
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_grade_"))
        zip_source_path = work_dir / "ref.zip"
        extract_folder = work_dir / "extract"
        extract_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(rag_zip_path, zip_source_path)
            if not zipfile.is_zipfile(zip_source_path):
                cleanup_grade_workspace(work_dir)
                return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
        except Exception as e:
            cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            try:
                rag_zip_path.unlink(missing_ok=True)
            except Exception:
                pass

    job_id = str(uuid.uuid4())
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    aup = (body.answer_user_prompt_text or "").strip()
    insert_fn = lambda rd, qa: update_rag_quiz_with_grade(
        rd,
        qa,
        rag_quiz_id=rag_quiz_id_int,
        answer_user_prompt_text=aup,
        quiz_content=qc_from_body,
    )
    background_tasks.add_task(
        run_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content_resolved,
        body.quiz_answer or "",
        _grade_job_results,
        insert_fn,
        aup,
        rag_quiz_id=rag_quiz_id_int,
        unit_type=grade_unit_type,
        transcription_grade=transcription_grade,
        quiz_user_prompt_text=quiz_user_prompt_db,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/for-exam
# ---------------------------------------------------------------------------

@router.post("/tab/unit/quiz/for-exam", summary="Set Rag Quiz for_exam flag")
def mark_rag_quiz_for_exam(body: RagQuizForExamRequest, caller_person_id: PersonId):
    """更新 Rag_Quiz.for_exam（true＝測驗用、false＝取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
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
        supabase.table("Rag_Quiz").update({"for_exam": body.for_exam, "updated_at": ts}).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()

        read = (
            supabase.table("Rag_Quiz")
            .select("*")
            .eq("rag_quiz_id", body.rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        row = (read.data or [{}])[0]
        return to_json_safe({
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
            "quiz_answer": row.get("answer_content") or row.get("quiz_answer"),
            "answer_critique": row.get("answer_critique"),
            "for_exam": row.get("for_exam"),
            "deleted": row.get("deleted"),
            "updated_at": row.get("updated_at"),
            "created_at": row.get("created_at"),
        })
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /rag/tab/unit/quiz/for-exam 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /rag/tab/unit/quiz/grade-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/tab/unit/quiz/grade-result/{job_id}", summary="Get Grade Result", tags=["rag"])
async def get_grade_result(job_id: str, _person_id: PersonId):
    """
    輪詢評分結果。status: pending | ready | error；
    ready 時 result 為 quiz_comments、rag_quiz_id（另含 rag_answer_id），並自資料庫讀取 rag_quiz 整列。
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
    out: dict[str, Any] = {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
    }
    rag_quiz_row: dict[str, Any] | None = None
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            rid = res.get("rag_quiz_id") or res.get("rag_answer_id")
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
        out["rag_quiz"] = rag_quiz_row
    return out


# ---------------------------------------------------------------------------
# GET /rag/transcript/text
# ---------------------------------------------------------------------------

@router.get("/transcript/text", response_model=RagTranscriptMarkdownResponse)
def rag_transcript_text(
    caller_person_id: PersonId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須恰好一個文字檔（.md .txt .doc .docx），回傳正文為 markdown",
    ),
):
    """自 upload ZIP 之 folder_name 路徑下讀取唯一一個文字檔的全文；markdown 僅為檔案內容，不寫回資料庫。"""
    _require_rag_tab_owner(caller_person_id, rag_tab_id)
    try:
        zip_bytes = read_upload_zip_bytes(caller_person_id, rag_tab_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    try:
        body, _path = read_single_transcript_text_from_upload_zip(zip_bytes, folder_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RagTranscriptMarkdownResponse(markdown=body if body else "")


# ---------------------------------------------------------------------------
# GET /rag/unit/audio-file
# ---------------------------------------------------------------------------


@router.get("/unit/audio-file")
def rag_unit_audio_file(
    caller_person_id: PersonId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同（與 GET /rag/transcript/audio 一致）",
    ),
):
    """
    自 upload ZIP 內指定資料夾擷取第一個支援的音訊檔，**不**經 Deepgram，直接回傳二進位內容。
    前端可將完整 URL（含 query `person_id`）設為 `<audio src>`。
    """
    _require_rag_tab_owner(caller_person_id, rag_tab_id)
    try:
        zip_bytes = read_upload_zip_bytes(caller_person_id, rag_tab_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip(zip_bytes, folder_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    media = audio_media_type_for_suffix(suffix)
    disp_name = Path(inner_path).name
    safe_disp = "".join(c if 32 <= ord(c) < 127 and c not in '\\"' else "_" for c in disp_name) or "audio"
    return Response(
        content=contents,
        media_type=media,
        headers={"Content-Disposition": f'inline; filename="{safe_disp}"'},
    )


# ---------------------------------------------------------------------------
# GET /rag/transcript/audio
# ---------------------------------------------------------------------------

@router.get("/transcript/audio", response_model=RagTranscriptMarkdownResponse)
def rag_transcript_audio(
    caller_person_id: PersonId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id（對應 Storage 路徑 {person_id}/{rag_tab_id}/upload/…）"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須有音訊檔。若僅有文字檔（內含 YouTube 連結）請改呼叫 GET /rag/transcript/youtube",
    ),
):
    """自 upload ZIP 內 folder_name 路徑找音訊檔，Deepgram 轉文字；markdown 僅為逐字稿正文。"""
    _require_rag_tab_owner(caller_person_id, rag_tab_id)
    try:
        zip_bytes = read_upload_zip_bytes(caller_person_id, rag_tab_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    try:
        contents, suffix, _inner_path = pick_audio_from_upload_zip(zip_bytes, folder_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    dg_model = (os.environ.get("DEEPGRAM_MODEL") or "nova-2").strip()
    try:
        text, elapsed = transcribe_audio_bytes_deepgram(
            contents,
            suffix=suffix or ".mp3",
            model=dg_model,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("GET /rag/transcript/audio Deepgram 錯誤")
        raise HTTPException(status_code=500, detail=f"轉錄失敗: {e!s}") from e

    return RagTranscriptMarkdownResponse(markdown=(text or "").strip())


# ---------------------------------------------------------------------------
# GET /rag/transcript/youtube
# ---------------------------------------------------------------------------

@router.get("/transcript/youtube", response_model=RagTranscriptMarkdownResponse)
def rag_transcript_youtube(
    caller_person_id: PersonId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須恰好一個文字檔（.md .txt .doc .docx，內含 YouTube 連結或 video_id），擷取 en 字幕為正文",
    ),
):
    """自 upload ZIP 內 folder_name 下唯一文字檔讀取連結，擷取 en 字幕；markdown 僅為字幕合併文字。"""
    _require_rag_tab_owner(caller_person_id, rag_tab_id)
    try:
        zip_bytes = read_upload_zip_bytes(caller_person_id, rag_tab_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    try:
        vid, _text_path = read_youtube_video_id_from_upload_zip(zip_bytes, folder_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        text, _elapsed = youtube_transcript_plain_text(vid, languages=["en"])
        return RagTranscriptMarkdownResponse(markdown=(text or "").strip())
    except InvalidVideoId as e:
        raise HTTPException(status_code=400, detail=youtube_transcript_api_user_message(e)) from e
    except (VideoUnavailable, NoTranscriptFound, TranscriptsDisabled) as e:
        raise HTTPException(status_code=404, detail=youtube_transcript_api_user_message(e)) from e
    except YouTubeTranscriptApiException as e:
        raise HTTPException(status_code=502, detail=youtube_transcript_api_user_message(e)) from e
    except Exception as e:
        _logger.exception("GET /rag/transcript/youtube 錯誤")
        raise HTTPException(status_code=500, detail=f"擷取字幕失敗: {e!s}") from e
