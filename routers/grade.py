"""
RAG 評分與出題 API 模組。
- POST /rag/tab/unit/quiz/llm-generate：依 rag_quiz_id 出題（LLM）；選填 quiz_history_list 避免重複出題；unit_type 1 僅 RAG ZIP 向量檢索；2/3/4 以 transcription 純生成；其餘載 RAG ZIP 向量檢索。
- POST /rag/tab/unit/quiz/llm-generate-db：同 llm-generate，唯 body 不包含 quiz_user_prompt_text，一律沿用 Rag_Quiz 列上既有值。
- POST /rag/tab/unit/quiz/llm-generate-followup：接續出題；答不好追問弱點，答好則出新題；quiz_history_list 為先前問答（題幹＋作答）列表。
- POST /rag/tab/unit/quiz/llm-generate-followup-db：同 llm-generate-followup，唯 body 不包含 quiz_user_prompt_text。
- POST /rag/tab/unit/quiz/llm-grade：非同步 RAG+LLM 評分（body 以 rag_id 置頂；quiz_content 可空，自 Rag_Quiz 讀題幹）；answer_user_prompt_text 可空——空字串會寫入並覆蓋該列。回傳 202 + job_id，輪詢 GET /rag/tab/unit/quiz/grade-result/{job_id}。
- POST /rag/tab/unit/quiz/llm-grade-db：同 llm-grade，唯 body 不包含 answer_user_prompt_text，評分與寫回一律沿用 Rag_Quiz.answer_user_prompt_text（不論請求）。
- POST /rag/tab/unit/quiz/for-exam：更新 Rag_Quiz.for_exam（body `for_exam` 預設 true；false 取消測驗用）。
- GET /rag/tab/unit/quiz/grade-result/{job_id}：輪詢評分結果（ready 時含 rag_quiz 整列）。
- GET /rag/transcript/text、audio／youtube：`with_timestamps` 預設 true；``timestamp_merge_seconds`` 控制標記稀疏度（預設約每 10 秒一標；``0``＝每一段／字幕一行，最密）。
- GET /rag/unit/mp3-file：自 upload ZIP 依單元資料夾回傳原始音訊 bytes（供 `<audio src>`）；query 須含 person_id，且須為該 rag_tab_id 之 Rag 擁有者。
- GET /rag/unit/youtube-url：自 upload ZIP 依 folder_name 解析 YouTube 連結／video_id；query 須含 person_id，且須為該 rag_tab_id 之 Rag 擁有者；語意對齊存庫 transcript 規則，不依 rag_unit_id。
"""

import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from utils.quiz_generation import (
    generate_quiz,
    generate_quiz_followup,
    generate_quiz_followup_transcription_only,
    generate_quiz_transcription_only,
)
from utils.openapi_request_body import openapi_body

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from postgrest.exceptions import APIError
from dependencies.person_id import PersonId
from dependencies.course_id import CourseId
from fastapi.responses import JSONResponse, Response
from pydantic import AliasChoices, BaseModel, Field

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
from utils.rag_course_utils import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    require_rag_tab_owner,
    select_without_course_id_if_needed,
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
    """POST /rag/tab/unit/quiz/llm-generate；請求 body 欄位順序同 public.Rag_Quiz（rag_quiz_id, quiz_name, quiz_user_prompt_text），末欄 quiz_history_list 為 API 擴充。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[str] = Field(
        default_factory=list,
        description="已出過的題目題幹（字串陣列）；送入 LLM 出題 prompt，避免重複出題",
    )


class QuizGradeRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade 請求 body。
    欄位順序：Rag.rag_id → public.Rag_Quiz（rag_tab_id, rag_quiz_id, quiz_content, answer_user_prompt_text, answer_content／quiz_answer）。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_tab_id: str = Field("", description="選填；後端以 Rag.rag_tab_id 為準")
    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
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


class GenerateQuizDbOnlyRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate-db；欄位順序同 Rag_Quiz（rag_quiz_id, quiz_name），末欄 quiz_history_list 為 API 擴充。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[str] = Field(
        default_factory=list,
        description="已出過的題目題幹（字串陣列）；送入 LLM 出題 prompt，避免重複出題",
    )


class QuizHistoryPair(BaseModel):
    """先前問答一組：題幹、作答、參考答案與評閱（對齊 Rag_Quiz 欄位）。"""

    quiz_content: str = Field(..., description="先前題目題幹")
    answer_content: str = Field(
        ...,
        description="先前作答（學生答案）",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    quiz_answer_reference: str = Field(
        "",
        description="該題參考答案（對齊 Rag_Quiz.quiz_answer_reference）",
        validation_alias=AliasChoices(
            "quiz_answer_reference",
            "quiz_reference_answer",
            "reference_answer",
        ),
    )
    answer_critique: str = Field(
        "",
        description="該題評閱／批改評語（對齊 Rag_Quiz.answer_critique）",
        validation_alias=AliasChoices("answer_critique", "critique", "quiz_comments"),
    )


class GenerateQuizFollowupRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate-followup；末欄 quiz_history_list 為先前問答（一問一答一項）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[QuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答列表；每項含 quiz_content、answer_content、quiz_answer_reference、answer_critique",
    )


class GenerateQuizFollowupDbOnlyRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate-followup-db；不含 quiz_user_prompt_text。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[QuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答列表；每項含 quiz_content、answer_content、quiz_answer_reference、answer_critique",
    )


class QuizGradeDbOnlyRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade-db。
    欄位順序：Rag.rag_id → Rag_Quiz（rag_tab_id, rag_quiz_id, quiz_content, answer_content／quiz_answer）；不含 answer_user_prompt_text。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_tab_id: str = Field("", description="選填；後端以 Rag.rag_tab_id 為準")
    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
    quiz_content: str = Field(
        "",
        description="選填；非空時優先並可寫入 Rag_Quiz；空則自該 rag_quiz_id 之 Rag_Quiz.quiz_content 讀取（須於庫記憶中有題幹）",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Quiz.answer_content）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


class RagQuizForExamRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/for-exam：欄位順序同 public.Rag_Quiz（rag_quiz_id, rag_tab_id, rag_unit_id, for_exam）。
    以 rag_quiz_id 更新 Rag_Quiz.for_exam；若一併傳入 rag_tab_id／rag_unit_id（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_tab_id: str = Field("", description="選填；與資料列 rag_tab_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    for_exam: bool = Field(True, description="true：標記為測驗用；false：取消測驗用")


class RagTranscriptMarkdownResponse(BaseModel):
    """GET /rag/transcript/text、audio、youtube 共用回傳：markdown 僅為正文，無額外標題或 meta 區塊。"""

    markdown: str = Field(..., description="正文純文字或原檔內容（無 # Transcript 包裝）")


class RagUnitYoutubeUrlFromZipResponse(BaseModel):
    """GET /rag/unit/youtube-url：自 upload ZIP 解析之標準 watch URL。"""

    youtube_url: str = Field(..., description="https://www.youtube.com/watch?v=…")


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate
# ---------------------------------------------------------------------------

_RAG_LLM_GENERATE_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": ["先前已出過的題幹文字"],
}

_RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": [
        {
            "quiz_content": "先前題目題幹",
            "answer_content": "學生先前作答",
            "quiz_answer_reference": "參考答案全文",
            "answer_critique": "批改評語（指出答不好之處）",
        },
    ],
}


def _quiz_history_qa_dicts(pairs: list[QuizHistoryPair]) -> list[dict[str, str]]:
    return [
        {
            "quiz_content": p.quiz_content,
            "answer_content": p.answer_content,
            "quiz_answer_reference": p.quiz_answer_reference,
            "answer_critique": p.answer_critique,
        }
        for p in pairs
    ]


def _rag_llm_generate_quiz_impl(
    *,
    rag_quiz_id: int,
    quiz_name: str,
    quiz_user_prompt_text: str,
    caller_person_id: str,
    course_id: int,
    followup: bool,
    quiz_history_stems: list[str] | None = None,
    quiz_history_qa: list[QuizHistoryPair] | None = None,
):
    supabase = get_supabase()

    def build_quiz_sel(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Quiz",
            "rag_quiz_id, rag_tab_id, rag_unit_id, quiz_user_prompt_text, course_id",
            with_course_filter,
        )
        q = (
            supabase.table("Rag_Quiz")
            .select(cols)
            .eq("rag_quiz_id", rag_quiz_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    q_sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_sel, course_id)
    if not q_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={rag_quiz_id} 的 Rag_Quiz")
    q_row = q_sel.data[0]
    qup_body = (quiz_user_prompt_text or "").strip()
    qup_db = (q_row.get("quiz_user_prompt_text") or "").strip()
    qup_for_llm = qup_body or qup_db
    source_rag_unit_id = int(q_row.get("rag_unit_id") or 0)
    if source_rag_unit_id <= 0:
        raise HTTPException(status_code=400, detail="該 rag_quiz_id 對應的 rag_unit_id 無效")

    def fetch_unit_row(*, include_folder_combination: bool):
        def build(with_course_filter: bool):
            base_cols = (
                "rag_unit_id, rag_tab_id, unit_name, folder_combination, transcription, unit_type, course_id"
                if include_folder_combination
                else "rag_unit_id, rag_tab_id, unit_name, transcription, unit_type, course_id"
            )
            cols = select_without_course_id_if_needed("Rag_Unit", base_cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", source_rag_unit_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        return execute_with_course_id_fallback("Rag_Unit", build, course_id)

    try:
        unit_sel = fetch_unit_row(include_folder_combination=True)
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            unit_sel = fetch_unit_row(include_folder_combination=False)
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
    # rag_tab_id 以 Rag_Unit 為準（FK 綁 rag_unit_id）；Quiz 欄位為冗餘，過期時回寫
    if unit_rag_tab_id:
        rag_tab_id = unit_rag_tab_id
        if source_rag_tab_id != unit_rag_tab_id:
            try:
                supabase.table("Rag_Quiz").update(
                    {"rag_tab_id": unit_rag_tab_id, "updated_at": now_taipei_iso()}
                ).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
            except Exception as e:
                _logger.warning(
                    "Rag_Quiz rag_tab_id 與 Rag_Unit 不一致，回寫失敗 rag_quiz_id=%s: %s",
                    rag_quiz_id,
                    e,
                )
    else:
        rag_tab_id = source_rag_tab_id
    if not rag_tab_id:
        raise HTTPException(status_code=400, detail="無法由 rag_quiz_id 解析 rag_tab_id")

    rag_sel = (
        supabase.table("Rag")
        .select("rag_id, course_id")
        .eq("rag_tab_id", rag_tab_id)
        .eq("course_id", course_id)
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
        qa_dicts = _quiz_history_qa_dicts(quiz_history_qa or [])
        if unit_type_val in (2, 3, 4):
            if followup:
                result = generate_quiz_followup_transcription_only(
                    api_key=api_key,
                    transcription=transcription_text,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list=qa_dicts,
                )
            else:
                result = generate_quiz_transcription_only(
                    api_key=api_key,
                    transcription=transcription_text,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list=quiz_history_stems or [],
                )
        else:
            path = get_zip_path(rag_zip_tab_id)
            if not path or not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（rag_tab_id={rag_zip_tab_id}）",
                )
            if followup:
                result = generate_quiz_followup(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list=qa_dicts,
                )
            else:
                result = generate_quiz(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list=quiz_history_stems or [],
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
        result["rag_quiz_id"] = rag_quiz_id
        qup_stored = qup_body if qup_body else qup_db
        qts = now_taipei_iso()
        body_quiz_name = quiz_name.strip()
        resolved_quiz_name = body_quiz_name or (
            (stem or "").strip() or (unit_row.get("unit_name") or "").strip() or ""
        )
        result["quiz_name"] = resolved_quiz_name
        result["follow_up"] = followup
        quiz_update: dict[str, Any] = {
            "rag_tab_id": rag_tab_id,
            "quiz_name": resolved_quiz_name,
            "quiz_user_prompt_text": qup_stored,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_user_prompt_text": "",
            "answer_content": "",
            "follow_up": followup,
            "updated_at": qts,
        }
        try:
            supabase.table("Rag_Quiz").update(quiz_update).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        except Exception as e:
            _logger.error(
                "Rag_Quiz llm-generate 更新失敗 rag_quiz_id=%s: %s",
                rag_quiz_id,
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
            .eq("rag_quiz_id", rag_quiz_id)
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
                rag_quiz_id,
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


@router.post("/tab/unit/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def rag_llm_generate_quiz(
    body: openapi_body(GenerateQuizRequest, _RAG_LLM_GENERATE_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    Body：rag_quiz_id、quiz_name、quiz_user_prompt_text（可空字串）、quiz_history_list（選填；順序同 public.Rag_Quiz）；
    rag_tab_id／rag_unit_id 由後端依 rag_quiz_id 自資料庫帶入；quiz_user_prompt_text 空則自該列 Rag_Quiz 讀取。
    選填 `quiz_history_list`（字串陣列）：已出過的題目題幹，由 `utils.quiz_generation` 併入 user「已出過題目」區塊，避免重複出題。
    unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcription。
    unit_type 2／3／4 時不載入 RAG ZIP，改以逐字稿為 context；與 unit_type=1 共用 `SYSTEM_PROMPT_QUIZ`、`USER_PROMPT_COURSE` 與 `_generate_quiz_from_context`。
    出題成功後更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=false；並清空 answer_*）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history_stems=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate-followup
# ---------------------------------------------------------------------------


@router.post(
    "/tab/unit/quiz/llm-generate-followup",
    summary="Rag LLM Generate Follow-up Quiz",
    operation_id="rag_llm_generate_quiz_followup",
)
def rag_llm_generate_quiz_followup(
    body: openapi_body(GenerateQuizFollowupRequest, _RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    依先前問答接續出下一題：作答不佳則針對弱點追問；作答良好則改出新的不重複題目。
    Body 與 `llm-generate` 類似，但 `quiz_history_list` 為物件陣列，
    每項含 `quiz_content`、`answer_content`、`quiz_answer_reference`、`answer_critique`，一問一答一項。
    使用 `SYSTEM_PROMPT_QUIZ_FOLLOWUP`／`USER_PROMPT_COURSE_FOLLOWUP`。
    出題成功後同樣更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=true；並清空 answer_*）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history_qa=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate-db
# ---------------------------------------------------------------------------


@router.post(
    "/tab/unit/quiz/llm-generate-db",
    summary="Rag LLM Generate Quiz (stored quiz_user_prompt_text)",
    operation_id="rag_llm_generate_quiz_db_prompt",
)
def rag_llm_generate_quiz_db_prompt(
    body: openapi_body(
        GenerateQuizDbOnlyRequest,
        {"rag_quiz_id": 1, "quiz_name": "", "quiz_history_list": ["先前已出過的題幹文字"]},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`（行為等同傳空字串至 `llm-generate`）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history_stems=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate-followup-db
# ---------------------------------------------------------------------------


@router.post(
    "/tab/unit/quiz/llm-generate-followup-db",
    summary="Rag LLM Generate Follow-up Quiz (stored quiz_user_prompt_text)",
    operation_id="rag_llm_generate_quiz_followup_db_prompt",
)
def rag_llm_generate_quiz_followup_db_prompt(
    body: openapi_body(
        GenerateQuizFollowupDbOnlyRequest,
        {
            "rag_quiz_id": 1,
            "quiz_name": "",
            "quiz_history_list": [
                {
                    "quiz_content": "先前題目題幹",
                    "answer_content": "學生先前作答",
                    "quiz_answer_reference": "參考答案全文",
                    "answer_critique": "批改評語",
                },
            ],
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate-followup` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history_qa=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-grade
# ---------------------------------------------------------------------------


async def _enqueue_rag_llm_grade_job(
    background_tasks: BackgroundTasks,
    caller_person_id: str,
    course_id: int,
    *,
    rag_id_str: str,
    rag_quiz_id_str: str,
    qc_from_body: str,
    quiz_answer: str,
    answer_user_prompt_mode: Literal["from_request", "from_rag_quiz_row"],
    answer_user_prompt_from_request: str = "",
) -> JSONResponse:
    """將 RAG llm-grade 工作排入 BackgroundTasks；`grade-result` 輪詢鍵為記憶體 job_id。"""
    rag_id_str = (rag_id_str or "").strip()
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
    assert_row_course_id(row, course_id, "Rag")

    try:
        rag_quiz_id_int = int(rag_quiz_id_str.strip()) if rag_quiz_id_str.strip() else 0
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

    def build_grade_quiz_sel(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Quiz",
            "rag_unit_id, quiz_user_prompt_text, quiz_content, answer_user_prompt_text, course_id",
            with_course_filter,
        )
        q = (
            supabase.table("Rag_Quiz")
            .select(cols)
            .eq("rag_quiz_id", rag_quiz_id_int)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    rq_sel = execute_with_course_id_fallback("Rag_Quiz", build_grade_quiz_sel, course_id)
    if not rq_sel.data:
        return JSONResponse(status_code=404, content={"error": f"找不到 rag_quiz_id={rag_quiz_id_int} 的 Rag_Quiz"})
    rq_row = rq_sel.data[0]
    quiz_user_prompt_db = (rq_row.get("quiz_user_prompt_text") or "").strip()
    answer_user_prompt_db = (rq_row.get("answer_user_prompt_text") or "").strip()
    qc_from_body = (qc_from_body or "").strip()
    qc_from_db = (rq_row.get("quiz_content") or "").strip()
    quiz_content_resolved = qc_from_body or qc_from_db
    if not quiz_content_resolved:
        return JSONResponse(
            status_code=400,
            content={
                "error": "缺少測驗題幹：請於請求傳入 quiz_content，或先於該 Rag_Quiz 設定 quiz_content。",
            },
        )

    if answer_user_prompt_mode == "from_rag_quiz_row":
        aup = answer_user_prompt_db
    else:
        aup = (answer_user_prompt_from_request or "").strip()

    grade_unit_type = 0
    transcription_text = ""
    try:
        ruid_raw = rq_row.get("rag_unit_id")
        ruid_i = int(ruid_raw) if ruid_raw is not None else 0
    except (TypeError, ValueError):
        ruid_i = 0
    if ruid_i > 0:
        def build_grade_unit_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Unit",
                "unit_type, transcription, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", ruid_i)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        uu = execute_with_course_id_fallback("Rag_Unit", build_grade_unit_sel, course_id)
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
        quiz_answer or "",
        _grade_job_results,
        insert_fn,
        aup,
        rag_quiz_id=rag_quiz_id_int,
        unit_type=grade_unit_type,
        transcription_grade=transcription_grade,
        quiz_user_prompt_text=quiz_user_prompt_db,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


@router.post("/tab/unit/quiz/llm-grade", summary="Rag Grade Quiz")
async def grade_submission(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizGradeRequest,
        {
            "rag_id": "1",
            "rag_tab_id": "",
            "rag_quiz_id": "1",
            "quiz_content": "",
            "answer_user_prompt_text": "",
            "quiz_answer": "學生作答文字",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    非同步評分：Body 以 rag_id、rag_quiz_id 為核心；quiz_content 可省略（自 Rag_Quiz 讀）。
    `answer_user_prompt_text` 以請求為準（可空；空字串會寫入並覆蓋 Rag_Quiz 該列）。
    unit_type 2／3／4 時以 transcription 純 LLM 批改；其餘依 rag_id 載入 RAG ZIP。
    回傳 202 + job_id；輪詢 GET /rag/tab/unit/quiz/grade-result/{job_id}。
    """
    return await _enqueue_rag_llm_grade_job(
        background_tasks,
        caller_person_id,
        course_id,
        rag_id_str=body.rag_id,
        rag_quiz_id_str=body.rag_quiz_id,
        qc_from_body=body.quiz_content,
        quiz_answer=body.quiz_answer,
        answer_user_prompt_mode="from_request",
        answer_user_prompt_from_request=(body.answer_user_prompt_text or "").strip(),
    )


@router.post(
    "/tab/unit/quiz/llm-grade-db",
    summary="Rag Grade Quiz (stored answer_user_prompt_text)",
    operation_id="rag_llm_grade_quiz_db_prompt",
)
async def grade_submission_stored_answer_prompt(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizGradeDbOnlyRequest,
        {
            "rag_id": "1",
            "rag_tab_id": "",
            "rag_quiz_id": "1",
            "quiz_content": "",
            "quiz_answer": "學生作答文字",
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-grade` 相同，但請求不含 `answer_user_prompt_text`；
    評分時與寫回皆以 Rag_Quiz 該列既有之 `answer_user_prompt_text` 為準。
    """
    return await _enqueue_rag_llm_grade_job(
        background_tasks,
        caller_person_id,
        course_id,
        rag_id_str=body.rag_id,
        rag_quiz_id_str=body.rag_quiz_id,
        qc_from_body=body.quiz_content,
        quiz_answer=body.quiz_answer,
        answer_user_prompt_mode="from_rag_quiz_row",
    )


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/for-exam
# ---------------------------------------------------------------------------

@router.post("/tab/unit/quiz/for-exam", summary="Set Rag Quiz for_exam flag")
def mark_rag_quiz_for_exam(
    body: openapi_body(
        RagQuizForExamRequest,
        {"rag_quiz_id": 1, "rag_tab_id": "", "rag_unit_id": 0, "for_exam": True},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.for_exam（true＝測驗用、false＝取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
    req_tab = (body.rag_tab_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
    try:
        supabase = get_supabase()
        def build_for_exam_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", body.rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_for_exam_sel, course_id)
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
            "follow_up": row.get("follow_up"),
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
async def get_grade_result(job_id: str, _person_id: PersonId, course_id: CourseId):
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

                        def build_grade_result_sel(with_course_filter: bool):
                            q = (
                                supabase.table("Rag_Quiz")
                                .select("*")
                                .eq("rag_quiz_id", rid_int)
                                .eq("deleted", False)
                            )
                            if with_course_filter and course_id is not None:
                                q = q.eq("course_id", course_id)
                            return q.limit(1)

                        q = execute_with_course_id_fallback(
                            "Rag_Quiz", build_grade_result_sel, course_id
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
    course_id: CourseId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須恰好一個文字檔（.md .txt .doc .docx），回傳正文為 markdown",
    ),
):
    """自 upload ZIP 之 folder_name 路徑下讀取唯一一個文字檔的全文；markdown 僅為檔案內容，不寫回資料庫。"""
    require_rag_tab_owner(caller_person_id, rag_tab_id, course_id)
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
# GET /rag/unit/mp3-file
# ---------------------------------------------------------------------------


@router.get("/unit/mp3-file")
def rag_unit_audio_file(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同（與 GET /rag/transcript/audio 一致）",
    ),
):
    """
    自 upload ZIP 內指定資料夾擷取第一個支援的音訊檔，**不**經 Deepgram，直接回傳二進位內容。
    query 須含 `person_id`，且須與該 `rag_tab_id` 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_tab_id, course_id)
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
# GET /rag/unit/youtube-url
# ---------------------------------------------------------------------------


@router.get("/unit/youtube-url", response_model=RagUnitYoutubeUrlFromZipResponse)
def rag_unit_youtube_url(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同（與 GET /rag/transcript/youtube、GET /rag/unit/mp3-file 一致）",
    ),
):
    """
    自 upload ZIP 內指定資料夾讀取**恰好一個**文字檔（.md／.txt／.doc／.docx），解析 YouTube 連結或 video_id，
    回傳標準 `watch` URL（不擷取字幕）。query 須含 `person_id`，且須與該 rag_tab_id 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_tab_id, course_id)
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
        vid, _inner_path = read_youtube_video_id_from_upload_zip(zip_bytes, folder_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RagUnitYoutubeUrlFromZipResponse(youtube_url=f"https://www.youtube.com/watch?v={vid}")


# ---------------------------------------------------------------------------
# GET /rag/transcript/audio
# ---------------------------------------------------------------------------

@router.get("/transcript/audio", response_model=RagTranscriptMarkdownResponse)
def rag_transcript_audio(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id（對應 Storage 路徑 {person_id}/{rag_tab_id}/upload/…）"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須有音訊檔。若僅有文字檔（內含 YouTube 連結）請改呼叫 GET /rag/transcript/youtube",
    ),
    with_timestamps: bool = Query(
        True,
        description="true（預設）：分段時間標記（utterances／段落／字級備援）；false：單段全文",
    ),
    timestamp_merge_seconds: float | None = Query(
        None,
        ge=0,
        description=(
            "時間標記合併間隔（秒）：未傳時預設約 10（約每 10 秒或累積一段再換標），或由環境變數 TRANSCRIPT_TIMESTAMP_MERGE_SECONDS；"
            "0＝不按時間窗合併（一句／一 utterance／一字幕片段一行）；僅 with_timestamps=true 時生效"
        ),
    ),
):
    """自 upload ZIP 內 folder_name 路徑找音訊檔，Deepgram 轉文字；預設 markdown 含較稀疏的句首時間標記。"""
    require_rag_tab_owner(caller_person_id, rag_tab_id, course_id)
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
            with_timestamps=with_timestamps,
            timestamp_merge_seconds=timestamp_merge_seconds,
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
    course_id: CourseId,
    rag_tab_id: str = Query(..., description="Rag.rag_tab_id"),
    folder_name: str = Query(
        ...,
        description="ZIP 內單元資料夾名；該資料夾下須恰好一個文字檔（.md .txt .doc .docx，內含 YouTube 連結或 video_id），擷取字幕為正文",
    ),
    languages: str | None = Query(
        None,
        description="字幕語言代碼優先序，逗號分隔（如 en,zh-Hant）；未填則 YOUTUBE_TRANSCRIPT_LANGUAGES 或預設 en→中文→日韓",
    ),
    with_timestamps: bool = Query(
        True,
        description="true（預設）：含時間標記；false：各段空白接成單段全文",
    ),
    timestamp_merge_seconds: float | None = Query(
        None,
        ge=0,
        description=(
            "時間標記合併間隔（秒）：未傳時預設約 10，或由環境變數 TRANSCRIPT_TIMESTAMP_MERGE_SECONDS；"
            "0＝每一字幕片段一行；僅 with_timestamps=true 時生效"
        ),
    ),
):
    """自 upload ZIP 內 folder_name 下唯一文字檔讀取連結，擷取字幕；預設為較稀疏的時間標記。"""
    require_rag_tab_owner(caller_person_id, rag_tab_id, course_id)
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

    raw_langs = (languages or "").strip()
    lang_list: list[str] | None = (
        [x.strip() for x in raw_langs.split(",") if x.strip()] if raw_langs else None
    )

    try:
        text, _elapsed = youtube_transcript_plain_text(
            vid,
            languages=lang_list,
            with_timestamps=with_timestamps,
            timestamp_merge_seconds=timestamp_merge_seconds,
        )
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
