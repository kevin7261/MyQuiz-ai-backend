"""
RAG 出題、評分、題目標記與課程設定 API（路徑順序見 utils.openapi_order）。

**題目標記**（在 llm-generate 之前，與 Exam 題目 CRUD 區塊對齊）：
- POST /rag/tab/unit/quiz/followup：更新 follow_up
- POST /rag/tab/unit/quiz/for-exam：更新 for_exam

**出題**：POST …/llm-generate(-db) → POST …/llm-generate-followup(-db)

**評分**：POST …/llm-grade(-db) → GET …/grade-result/{job_id}

**單元資源**：GET /rag/unit/text、/rag/unit/mp3-file、/rag/unit/youtube-url

**設定**：GET/PUT /rag/llm_api_key、/rag/llm_model；個人／課程分析 prompt 見 course_settings 路由。
"""

import base64
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from services.quiz_generation import (
    format_quiz_history_prompt_for_llm,
    generate_quiz,
    generate_quiz_followup,
    generate_quiz_followup_transcript_only,
    generate_quiz_transcript_only,
)
from utils.openapi import openapi_body

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from postgrest.exceptions import APIError
from dependencies.person_id import PersonId
from dependencies.course_id import CourseId
from fastapi.responses import JSONResponse, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_rag_quiz_with_grade,
    _rag_quiz_missing_column_error,
)
from utils.taipei_time import now_taipei_iso
from utils.serialization import to_json_safe
from utils.llm_key import fetch_api_key_setting_row, get_rag_api_key, get_rag_llm_model
from utils.course_setting import COURSE_SETTING_RAG_API_KEY, COURSE_SETTING_LLM_MODEL
from routers.course_settings import (
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)
from utils.media import (
    audio_media_type_for_suffix,
)
from utils.rag_stem import get_rag_stem_from_rag_id, instruction_from_rag_row, transcript_from_row
from utils.rag_transcript import (
    pick_audio_from_upload_zip,
    read_mp3_unit_transcript_from_upload_zip,
    read_single_transcript_text_from_upload_zip,
    read_supplementary_text_from_youtube_unit,
    read_upload_zip_bytes,
    read_youtube_video_id_from_upload_zip,
)
from utils.rag_course import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    require_rag_tab_owner,
    resolve_rag_tab_owner_person_id,
    select_without_course_id_if_needed,
)
from utils.supabase import get_supabase
from utils.zip_storage import get_zip_path
from utils.db_schema import (
    QUIZ_HISTORY_OPENAPI_ITEM,
    QUIZ_HISTORY_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST,
    coerce_quiz_history_prompt_text_request,
    coerce_quiz_history_request,
    parse_quiz_history_prompt_text,
    parse_rag_quiz_history_list,
    resolve_quiz_history_for_generate,
    serialize_quiz_history_prompt_text,
    serialize_rag_quiz_history_list,
)

router = APIRouter(prefix="/rag", tags=["rag"])

_logger = logging.getLogger(__name__)

_grade_job_results: dict[str, dict[str, Any]] = {}

RAG_UNIT_TYPE_TEXT = 2


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QuizHistoryPair(BaseModel):
    """quiz_history_list 單筆：八欄位物件。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_OPENAPI_ITEM]},
    )

    rag_unit_id: int = Field(0, ge=0, description="Rag_Unit 主鍵", examples=[1])
    quiz_name: str = Field("", description="題型名稱")
    follow_up: bool = Field(
        False,
        description="是否為追問題",
        validation_alias=AliasChoices("follow_up", "followup", "followUp"),
    )
    quiz_content: str = Field(..., description="先前題目題幹")
    quiz_hint: str = Field("", description="提示")
    answer_content: str = Field(
        "",
        description="先前作答（學生答案）",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    quiz_answer_reference: str = Field(
        "",
        description="該題參考答案（對齊 quiz_answer_reference）",
        validation_alias=AliasChoices(
            "quiz_answer_reference",
            "quiz_reference_answer",
            "reference_answer",
        ),
    )
    answer_critique: str = Field(
        "",
        description="該題評閱／批改評語（對齊 answer_critique）",
        validation_alias=AliasChoices("answer_critique", "critique", "quiz_comments"),
    )


def _coerce_quiz_history_list_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list（不讀 DB）。"""
    return coerce_quiz_history_request(v)


def _coerce_quiz_history_prompt_stem_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（一般出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=False)


def _coerce_quiz_history_prompt_followup_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（追問出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=True)


_QUIZ_HISTORY_LIST_FIELD = Field(
    default_factory=list,
    description="先前問答（八欄位 JSON 物件陣列）；僅寫入 DB",
)
_QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD = Field(
    default_factory=list,
    description="併入 LLM 出題 prompt 的先前題幹（JSON 物件陣列，每筆僅 quiz_content）；寫入 DB",
)
_QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD = Field(
    default_factory=list,
    description=(
        "併入 LLM 追問 prompt 的先前問答（JSON 物件陣列：quiz_content、"
        "quiz_answer_reference、answer_content、answer_critique）；寫入 DB"
    ),
)


class QuizHistoryPromptStem(BaseModel):
    """quiz_history_list_prompt_text 單筆（一般出題）。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM]},
    )

    quiz_content: str = Field(..., description="先前題目題幹")


class QuizHistoryPromptFollowup(BaseModel):
    """quiz_history_list_prompt_text 單筆（追問出題）。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM]},
    )

    quiz_content: str = Field(..., description="先前題目題幹")
    quiz_answer_reference: str = Field("", description="參考答案全文")
    answer_content: str = Field(
        "",
        description="學生先前作答",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    answer_critique: str = Field("", description="批改評語")


class GenerateQuizRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate；含 quiz_history_list（八欄位物件陣列）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptStem] = _QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_stem_validator(v)


class QuizGradeRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade 請求 body。
    欄位順序：Rag.rag_id → public.Rag_Quiz（rag_page_id, rag_quiz_id, quiz_content, answer_user_prompt_text, answer_content／quiz_answer）。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_page_id: str = Field("", description="選填；後端以 Rag.rag_page_id 為準")
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
    """POST /rag/tab/unit/quiz/llm-generate-db；含 quiz_history_list（八欄位物件陣列）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptStem] = _QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_stem_validator(v)


class GenerateQuizFollowupRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate-followup；含 quiz_history_list（先前問答 JSON 陣列，對齊 DB 欄位）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptFollowup] = (
        _QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_followup_validator(v)


class GenerateQuizFollowupDbOnlyRequest(BaseModel):
    """POST /rag/tab/unit/quiz/llm-generate-followup-db；不含 quiz_user_prompt_text。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptFollowup] = (
        _QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_followup_validator(v)


class QuizGradeDbOnlyRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/llm-grade-db。
    欄位順序：Rag.rag_id → Rag_Quiz（rag_page_id, rag_quiz_id, quiz_content, answer_content／quiz_answer）；不含 answer_user_prompt_text。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_page_id: str = Field("", description="選填；後端以 Rag.rag_page_id 為準")
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
    POST /rag/tab/unit/quiz/for-exam：欄位順序同 public.Rag_Quiz（rag_quiz_id, rag_page_id, rag_unit_id, for_exam）。
    以 rag_quiz_id 更新 Rag_Quiz.for_exam；若一併傳入 rag_page_id／rag_unit_id（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_page_id: str = Field("", description="選填；與資料列 rag_page_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    for_exam: bool = Field(True, description="true：標記為測驗用；false：取消測驗用")


class RagQuizFollowupRequest(BaseModel):
    """
    POST /rag/tab/unit/quiz/followup：欄位順序同 public.Rag_Quiz（rag_quiz_id, rag_page_id, rag_unit_id, follow_up）。
    以 rag_quiz_id 更新 Rag_Quiz.follow_up；若一併傳入 rag_page_id／rag_unit_id（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_page_id: str = Field("", description="選填；與資料列 rag_page_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    followup: bool = Field(
        False,
        description="true：標記為追問題；false：取消追問標記",
        validation_alias=AliasChoices("followup", "follow_up", "followUp"),
    )


class RagUnitTextResponse(BaseModel):
    """GET /rag/unit/text 回應。"""

    rag_page_id: str
    folder_name: str = ""
    rag_unit_id: int = 0
    text_file_name: str = ""
    transcript: str = ""


class RagUnitMp3FileFromZipResponse(BaseModel):
    """GET /rag/unit/mp3-file：自 upload ZIP 擷取之音訊與同資料夾文字檔逐字稿。"""

    rag_page_id: str
    folder_name: str
    audio_base64: str
    media_type: str
    filename: str
    text_file_name: str = ""
    transcript: str = ""


class RagUnitYoutubeUrlFromZipResponse(BaseModel):
    """GET /rag/unit/youtube-url：自 upload ZIP 解析 watch URL 與文字檔第二行起逐字稿。"""

    rag_page_id: str
    folder_name: str
    youtube_url: str = Field(..., description="https://www.youtube.com/watch?v=…")
    text_file_name: str = ""
    transcript: str = ""


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/followup
# ---------------------------------------------------------------------------


@router.post("/tab/unit/quiz/followup", summary="Set Rag Quiz follow_up flag", operation_id="rag_quiz_followup")
def mark_rag_quiz_followup(
    body: openapi_body(
        RagQuizFollowupRequest,
        {"rag_quiz_id": 1, "rag_page_id": "", "rag_unit_id": 0, "followup": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.follow_up（followup=true 標記追問、false 取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
    req_tab = (body.rag_page_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
    try:
        supabase = get_supabase()

        def build_followup_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
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

        sel = execute_with_course_id_fallback("Rag_Quiz", build_followup_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz，或已刪除")

        row0 = sel.data[0]
        pid = (row0.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權更新該 Rag_Quiz")
        if req_tab and (row0.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_quiz_id 對應資料不一致")
        if req_unit > 0 and int(row0.get("rag_unit_id") or 0) != req_unit:
            raise HTTPException(status_code=400, detail="rag_unit_id 與 rag_quiz_id 對應資料不一致")

        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update(
            {"follow_up": body.followup, "updated_at": ts}
        ).eq("rag_quiz_id", body.rag_quiz_id).eq("deleted", False).execute()

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
            "rag_page_id": row.get("rag_page_id"),
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
        _logger.exception("POST /rag/tab/unit/quiz/followup 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /rag/tab/unit/quiz/llm-generate
# ---------------------------------------------------------------------------

_RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE = list(QUIZ_HISTORY_PROMPT_STEM_OPENAPI_LIST)
_RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE = list(QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_LIST)

_RAG_LLM_GENERATE_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_user_prompt_text": "",
    "quiz_history_list": [
        {**QUIZ_HISTORY_OPENAPI_ITEM, "answer_critique": "批改評語（指出答不好之處）"},
    ],
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}

_RAG_LLM_GENERATE_DB_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_history_list": list(QUIZ_HISTORY_OPENAPI_LIST),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_STEM_EXAMPLE,
}

_RAG_LLM_GENERATE_FOLLOWUP_DB_OPENAPI_EXAMPLE = {
    "rag_quiz_id": 1,
    "quiz_name": "",
    "quiz_history_list": list(_RAG_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLE["quiz_history_list"]),
    "quiz_history_list_prompt_text": _RAG_QUIZ_HISTORY_PROMPT_FOLLOWUP_EXAMPLE,
}


def _safe_unlink(p: Path) -> None:
    """刪除暫存檔；忽略檔案不存在或刪除失敗。"""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _quiz_history_qa_dicts(pairs: list[QuizHistoryPair]) -> list[dict[str, Any]]:
    return parse_rag_quiz_history_list([p.model_dump() for p in pairs])


def _quiz_history_prompt_dicts(
    pairs: list[QuizHistoryPromptStem] | list[QuizHistoryPromptFollowup],
    *,
    followup: bool,
) -> list[dict[str, Any]]:
    return parse_quiz_history_prompt_text([p.model_dump() for p in pairs], followup=followup)


def _resolve_rag_quiz_page_id(
    supabase: Any, *, unit_rag_page_id: str, source_rag_page_id: str, rag_quiz_id: int
) -> str:
    """rag_page_id 以 Rag_Unit 為準（FK 綁 rag_unit_id）；Quiz 欄位為冗餘，過期時回寫。回傳解析後的 rag_page_id。"""
    if unit_rag_page_id:
        rag_page_id = unit_rag_page_id
        if source_rag_page_id != unit_rag_page_id:
            try:
                supabase.table("Rag_Quiz").update(
                    {"rag_page_id": unit_rag_page_id, "updated_at": now_taipei_iso()}
                ).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
            except Exception as e:
                _logger.warning(
                    "Rag_Quiz rag_page_id 與 Rag_Unit 不一致，回寫失敗 rag_quiz_id=%s: %s",
                    rag_quiz_id,
                    e,
                )
    else:
        rag_page_id = source_rag_page_id
    if not rag_page_id:
        raise HTTPException(status_code=400, detail="無法由 rag_quiz_id 解析 rag_page_id")
    return rag_page_id


def _prewrite_rag_quiz_history_fields(
    supabase: Any,
    *,
    rag_quiz_id: int,
    qa_dicts: list[dict[str, Any]],
    quiz_history_list_prompt_text: str,
) -> None:
    """出題前寫入 quiz_history_list 與 quiz_history_list_prompt_text（皆為 JSON 字串）。"""
    payload: dict[str, Any] = {
        "quiz_history_list": serialize_rag_quiz_history_list(qa_dicts),
        "quiz_history_list_prompt_text": quiz_history_list_prompt_text or "[]",
        "updated_at": now_taipei_iso(),
    }
    try:
        for _ in range(4):
            try:
                supabase.table("Rag_Quiz").update(payload).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
                return
            except Exception as upd_err:
                if _rag_quiz_missing_column_error(upd_err, "quiz_history_list") and "quiz_history_list" in payload:
                    payload.pop("quiz_history_list")
                    continue
                if _rag_quiz_missing_column_error(upd_err, "quiz_history_list_prompt_text") and "quiz_history_list_prompt_text" in payload:
                    payload.pop("quiz_history_list_prompt_text")
                    continue
                raise
    except Exception as e:
        _logger.warning(
            "Rag_Quiz 預寫 quiz_history 欄位略過 rag_quiz_id=%s: %s", rag_quiz_id, e
        )


def _persist_and_verify_rag_quiz(
    supabase: Any, *, rag_quiz_id: int, quiz_update: dict[str, Any], qc: str
) -> None:
    """更新 Rag_Quiz 出題結果並讀回驗證；任何失敗或讀回不一致皆拋 500 HTTPException。"""
    update_payload = dict(quiz_update)
    try:
        for _ in range(4):
            try:
                supabase.table("Rag_Quiz").update(update_payload).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
                break
            except Exception as upd_err:
                if _rag_quiz_missing_column_error(upd_err, "quiz_history_list") and "quiz_history_list" in update_payload:
                    update_payload.pop("quiz_history_list")
                    continue
                if _rag_quiz_missing_column_error(upd_err, "quiz_history_list_prompt_text") and "quiz_history_list_prompt_text" in update_payload:
                    update_payload.pop("quiz_history_list_prompt_text")
                    continue
                if _rag_quiz_missing_column_error(upd_err, "quiz_llm_model") and "quiz_llm_model" in update_payload:
                    update_payload.pop("quiz_llm_model")
                    continue
                raise
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


def _rag_llm_generate_quiz_impl(
    *,
    rag_quiz_id: int,
    quiz_name: str,
    quiz_user_prompt_text: str,
    caller_person_id: str,
    course_id: int,
    followup: bool,
    quiz_history: list[QuizHistoryPair] | None = None,
    quiz_history_list_prompt_items: list[dict[str, Any]] | None = None,
):
    supabase = get_supabase()

    def build_quiz_sel(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Quiz",
            "rag_quiz_id, rag_page_id, rag_unit_id, quiz_user_prompt_text, quiz_history_list, course_id",
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

    try:
        q_sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_sel, course_id)
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "quiz_history_list" in msg:
            def build_quiz_sel_no_history(with_course_filter: bool):
                cols = select_without_course_id_if_needed(
                    "Rag_Quiz",
                    "rag_quiz_id, rag_page_id, rag_unit_id, quiz_user_prompt_text, course_id",
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

            q_sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_sel_no_history, course_id)
        else:
            raise
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
                "rag_unit_id, rag_page_id, unit_name, folder_combination, transcript, unit_type, course_id"
                if include_folder_combination
                else "rag_unit_id, rag_page_id, unit_name, transcript, unit_type, course_id"
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
    unit_rag_page_id = (unit_row.get("rag_page_id") or "").strip()
    source_rag_page_id = (q_row.get("rag_page_id") or "").strip()
    rag_page_id = _resolve_rag_quiz_page_id(
        supabase,
        unit_rag_page_id=unit_rag_page_id,
        source_rag_page_id=source_rag_page_id,
        rag_quiz_id=rag_quiz_id,
    )

    rag_sel = (
        supabase.table("Rag")
        .select("rag_id, course_id")
        .eq("rag_page_id", rag_page_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_sel.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_page_id={rag_page_id} 的 Rag")
    rag_id = int(rag_sel.data[0].get("rag_id") or 0)
    if rag_id <= 0:
        raise HTTPException(status_code=400, detail="該 rag_page_id 對應的 rag_id 無效")

    row, stem, rag_zip_page_id = get_rag_stem_from_rag_id(
        supabase,
        rag_id,
        include_row=True,
        unit_name=unit_filter,
        rag_unit_id=source_rag_unit_id,
    )
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該筆 Rag 的 person_id 為空，無法出題")
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Rag 出題")
    api_key = get_rag_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 RAG API Key：PUT /rag/llm_api_key（Course_Setting key=rag-api-key，依 course_id）",
        )
    llm_model = get_rag_llm_model(course_id)
    transcript_text = transcript_from_row(unit_row)
    if not transcript_text:
        transcript_text = instruction_from_rag_row(row)

    try:
        unit_type_val = int(unit_row.get("unit_type") or 0)
    except (TypeError, ValueError):
        unit_type_val = 0

    if unit_type_val in (2, 3, 4) and not transcript_text:
        raise HTTPException(
            status_code=400,
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 或 Rag 設定 transcript，或經 POST /rag/tab/build-rag-zip 寫入 Rag_Unit.transcript",
        )

    path: Path | None = None
    try:
        request_history = _quiz_history_qa_dicts(quiz_history or [])
        qa_dicts, _stems_for_llm = resolve_quiz_history_for_generate(
            request_history=request_history,
        )
        prompt_dicts = list(quiz_history_list_prompt_items or [])
        prompt_db_str = serialize_quiz_history_prompt_text(prompt_dicts, followup=followup)
        prompt_for_llm = format_quiz_history_prompt_for_llm(prompt_dicts, followup=followup)
        _prewrite_rag_quiz_history_fields(
            supabase,
            rag_quiz_id=rag_quiz_id,
            qa_dicts=qa_dicts,
            quiz_history_list_prompt_text=prompt_db_str,
        )
        if unit_type_val in (2, 3, 4):
            if followup:
                result = generate_quiz_followup_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                )
            else:
                result = generate_quiz_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                )
        else:
            path = get_zip_path(rag_zip_page_id)
            if not path or not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（rag_page_id={rag_zip_page_id}）",
                )
            if followup:
                result = generate_quiz_followup(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                )
            else:
                result = generate_quiz(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=qup_for_llm,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                )
        result["transcript"] = "" if unit_type_val == 1 else transcript_text
        result["rag_output"] = {
            "rag_page_id": stem,
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
        result["quiz_llm_model"] = llm_model
        if qa_dicts:
            result["quiz_history_list"] = qa_dicts
        result["quiz_history_list_prompt_text"] = prompt_dicts
        quiz_update: dict[str, Any] = {
            "rag_page_id": rag_page_id,
            "quiz_name": resolved_quiz_name,
            "quiz_user_prompt_text": qup_stored,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_content": None,
            "answer_critique": None,
            "follow_up": followup,
            "quiz_history_list": serialize_rag_quiz_history_list(qa_dicts),
            "quiz_history_list_prompt_text": prompt_db_str,
            "quiz_llm_model": llm_model,
            "updated_at": qts,
        }
        _persist_and_verify_rag_quiz(
            supabase, rag_quiz_id=rag_quiz_id, quiz_update=quiz_update, qc=qc
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
            _safe_unlink(path)


@router.post("/tab/unit/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="rag_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def rag_llm_generate_quiz(
    body: openapi_body(GenerateQuizRequest, _RAG_LLM_GENERATE_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    Body：rag_quiz_id、quiz_name、quiz_user_prompt_text（可空字串）、quiz_history_list（選填；對齊 public.Rag_Quiz 欄位）；
    rag_page_id／rag_unit_id 由後端依 rag_quiz_id 自資料庫帶入；quiz_user_prompt_text 空則自該列 Rag_Quiz 讀取。
    選填 `quiz_history_list`（八欄位 JSON 物件陣列）：僅寫入 DB。
    選填 `quiz_history_list_prompt_text`（JSON 物件陣列，每筆僅 quiz_content）：併入 LLM 出題 prompt；寫入 DB。
    unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcript。
    unit_type 2／3／4 時不載入 RAG ZIP，改以逐字稿為 context；與 unit_type=1 共用 `SYSTEM_PROMPT_QUIZ`、`USER_PROMPT_COURSE` 與 `_generate_quiz_from_context`。
    出題成功後更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=false、quiz_history_list；清空 answer_content、answer_critique；保留 answer_user_prompt_text）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
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
    Body 與 `llm-generate` 類似：`quiz_history_list` 僅寫 DB；`quiz_history_list_prompt_text`（四欄位 JSON 物件陣列）併入 LLM prompt。
    `quiz_history_list` 為八欄位物件陣列（見 OpenAPI Example）。
    使用 `SYSTEM_PROMPT_QUIZ_FOLLOWUP`／`USER_PROMPT_COURSE_FOLLOWUP`。
    出題成功後同樣更新 public.Rag_Quiz（quiz_name、quiz_*、follow_up=true；寫入 quiz_history_list 為請求或 DB 既有之先前問答；清空 answer_content、answer_critique；保留 answer_user_prompt_text）。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text=body.quiz_user_prompt_text,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
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
    body: openapi_body(GenerateQuizDbOnlyRequest, _RAG_LLM_GENERATE_DB_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`（行為等同傳空字串至 `llm-generate`）。
    出題成功後清空 answer_content、answer_critique；保留 answer_user_prompt_text。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=False,
        ),
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
    body: openapi_body(GenerateQuizFollowupDbOnlyRequest, _RAG_LLM_GENERATE_FOLLOWUP_DB_OPENAPI_EXAMPLE),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    與 `llm-generate-followup` 相同，但請求不含 `quiz_user_prompt_text`，出題時一律使用
    Rag_Quiz 該列既有之 `quiz_user_prompt_text`。
    出題成功後清空 answer_content、answer_critique；保留 answer_user_prompt_text。
    """
    return _rag_llm_generate_quiz_impl(
        rag_quiz_id=body.rag_quiz_id,
        quiz_name=body.quiz_name,
        quiz_user_prompt_text="",
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history=body.quiz_history_list,
        quiz_history_list_prompt_items=_quiz_history_prompt_dicts(
            body.quiz_history_list_prompt_text,
            followup=True,
        ),
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
        row, stem, rag_zip_page_id = get_rag_stem_from_rag_id(supabase, rag_id_int, include_row=True)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        return JSONResponse(status_code=400, content={"error": "該筆 Rag 的 person_id 為空，無法評分"})
    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Rag 評分"})
    assert_row_course_id(row, course_id, "Rag")

    try:
        rag_quiz_id_int = int(rag_quiz_id_str.strip()) if rag_quiz_id_str.strip() else 0
    except ValueError:
        rag_quiz_id_int = 0
    if rag_quiz_id_int <= 0:
        return JSONResponse(status_code=400, content={"error": "rag_quiz_id 必填且須為大於 0 的整數（對應 Rag_Quiz 主鍵）"})

    api_key = get_rag_api_key(course_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "請設定 RAG API Key：PUT /rag/llm_api_key（Course_Setting key=rag-api-key，依 course_id）",
            },
        )
    llm_model = get_rag_llm_model(course_id)

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
    transcript_text = ""
    try:
        ruid_raw = rq_row.get("rag_unit_id")
        ruid_i = int(ruid_raw) if ruid_raw is not None else 0
    except (TypeError, ValueError):
        ruid_i = 0
    if ruid_i > 0:
        def build_grade_unit_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Unit",
                "unit_type, transcript, course_id",
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
            transcript_text = transcript_from_row(u0)
    if not transcript_text:
        transcript_text = instruction_from_rag_row(row)

    transcript_grade: str | None = None

    if grade_unit_type in (2, 3, 4):
        if not transcript_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcript 未設定：請於 Rag_Unit 或 Rag 設定 transcript（單元 2／3／4）"},
            )
        transcript_grade = transcript_text
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_grade_tx_"))
    else:
        rag_zip_path = get_zip_path(rag_zip_page_id)
        if not rag_zip_path or not rag_zip_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"找不到 RAG ZIP，請確認 rag_id={rag_id_str}（page_id={rag_zip_page_id}）"},
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
            _safe_unlink(rag_zip_path)

    job_id = str(uuid.uuid4())
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    def insert_fn(rd, qa):
        return update_rag_quiz_with_grade(
            rd,
            qa,
            rag_quiz_id=rag_quiz_id_int,
            answer_user_prompt_text=aup,
            quiz_content=qc_from_body,
            grade_llm_model=llm_model,
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
        transcript_grade=transcript_grade,
        quiz_user_prompt_text=quiz_user_prompt_db,
        llm_model=llm_model,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "grade_llm_model": llm_model})


@router.post("/tab/unit/quiz/llm-grade", summary="Rag Grade Quiz")
async def grade_submission(
    background_tasks: BackgroundTasks,
    body: openapi_body(
        QuizGradeRequest,
        {
            "rag_id": "1",
            "rag_page_id": "",
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
    unit_type 2／3／4 時以 transcript 純 LLM 批改；其餘依 rag_id 載入 RAG ZIP。
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
            "rag_page_id": "",
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
        {"rag_quiz_id": 1, "rag_page_id": "", "rag_unit_id": 0, "for_exam": True},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新 Rag_Quiz.for_exam（true＝測驗用、false＝取消）。以 rag_quiz_id 定位；僅 deleted=false 且 person_id 一致者可更新。"""
    req_tab = (body.rag_page_id or "").strip()
    req_unit = int(body.rag_unit_id or 0)
    try:
        supabase = get_supabase()
        def build_for_exam_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
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
        if req_tab and (row0.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_quiz_id 對應資料不一致")
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
            "rag_page_id": row.get("rag_page_id"),
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
                            if isinstance(rag_quiz_row, dict):
                                rag_quiz_row["quiz_history_list"] = parse_rag_quiz_history_list(
                                    rag_quiz_row.get("quiz_history_list")
                                )
                except (TypeError, ValueError) as e:
                    _logger.debug("grade-result rag_quiz_id 無效 job_id=%s: %s", job_id, e)
                except Exception as e:
                    _logger.warning("grade-result 讀取 Rag_Quiz 失敗 job_id=%s: %s", job_id, e)
        out["rag_quiz"] = rag_quiz_row
    return out


# ---------------------------------------------------------------------------
# GET /rag/unit/text
# ---------------------------------------------------------------------------


def _read_upload_zip_bytes_or_http_error(person_id: str, rag_page_id: str) -> bytes:
    """讀取 upload ZIP 內容；對應 404（找不到）／400（值錯誤）／500（其他）HTTPException。"""
    try:
        return read_upload_zip_bytes(person_id, rag_page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e


def _transcript_from_upload_zip_for_folder(
    person_id: str,
    rag_page_id: str,
    folder_name: str,
) -> tuple[str, str]:
    """自 upload ZIP 資料夾讀取 unit_type=2 文字檔全文；回傳 (transcript, text_file_name)。"""
    zip_bytes = read_upload_zip_bytes(person_id, rag_page_id)
    text, inner_path = read_single_transcript_text_from_upload_zip(zip_bytes, folder_name)
    return text, Path(inner_path).name


@router.get(
    "/unit/text",
    summary="Rag Unit Text",
    operation_id="rag_unit_text",
    response_model=RagUnitTextResponse,
)
def rag_unit_text(
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab）"),
    folder_name: str = Query(
        "",
        description="與 upload ZIP 內單元資料夾名相同；與 rag_unit_id 二擇一（有 folder_name 時須傳 person_id）",
    ),
    rag_unit_id: int = Query(
        0,
        ge=0,
        description="Rag_Unit 主鍵；與 folder_name 二擇一",
    ),
    person_id: Annotated[
        str | None,
        Query(
            alias="person_id",
            description="使用 folder_name 時必填；僅 rag_unit_id 時可不傳",
        ),
    ] = None,
):
    """
    回傳 **unit_type=2（文字單元）** 之 `text_file_name` 與 `transcript`（全文，含 Markdown）。

    - **folder_name**：自 upload ZIP 讀取（與 build-rag-zip unit_type=2 一致）；須傳 `person_id`。
    - **rag_unit_id**：自 `Rag_Unit` 讀取，**不需** `person_id`；若 DB 無逐字稿則改讀 upload ZIP（以 `folder_combination` 或 `unit_name` 為資料夾名）。
    """
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    unit_id = int(rag_unit_id or 0)

    if folder and unit_id > 0:
        raise HTTPException(status_code=400, detail="folder_name 與 rag_unit_id 請二擇一")
    if not folder and unit_id <= 0:
        raise HTTPException(status_code=400, detail="請傳入 folder_name 或 rag_unit_id（二擇一）")

    if folder:
        pid = (person_id or "").strip()
        if not pid:
            raise HTTPException(status_code=400, detail="使用 folder_name 時須傳入 person_id")
        require_rag_tab_owner(pid, rag_page_id, course_id)
        try:
            transcript, text_file_name = _transcript_from_upload_zip_for_folder(pid, tab, folder)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _logger.exception("GET /rag/unit/text 讀取 upload ZIP 失敗")
            raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e
        return RagUnitTextResponse(
            rag_page_id=tab,
            folder_name=folder,
            rag_unit_id=0,
            text_file_name=text_file_name,
            transcript=transcript,
        )

    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    supabase = get_supabase()

    def build_text_sel(with_course_filter: bool, *, include_folder: bool):
        cols = (
            "rag_unit_id, rag_page_id, unit_type, unit_name, folder_combination, "
            "text_file_name, transcript, deleted, course_id"
            if include_folder
            else "rag_unit_id, rag_page_id, unit_type, unit_name, text_file_name, transcript, deleted, course_id"
        )
        cols = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
        q = (
            supabase.table("Rag_Unit")
            .select(cols)
            .eq("rag_unit_id", unit_id)
            .eq("person_id", owner_pid)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    try:
        sel = execute_with_course_id_fallback(
            "Rag_Unit",
            lambda wc: build_text_sel(wc, include_folder=True),
            course_id,
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            try:
                sel = execute_with_course_id_fallback(
                    "Rag_Unit",
                    lambda wc: build_text_sel(wc, include_folder=False),
                    course_id,
                )
            except Exception as e2:
                _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
                raise HTTPException(status_code=500, detail=f"查詢失敗: {e2!s}") from e2
        else:
            _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
            raise HTTPException(status_code=500, detail=f"查詢失敗: {e!s}") from e
    except Exception as e:
        _logger.exception("GET /rag/unit/text 查詢 Rag_Unit 失敗")
        raise HTTPException(status_code=500, detail=f"查詢失敗: {e!s}") from e

    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_unit_id，或與此 rag_page_id／擁有者不一致",
        )
    row = sel.data[0]
    if row.get("deleted"):
        raise HTTPException(status_code=404, detail="該單元已刪除")
    if (row.get("rag_page_id") or "").strip() != tab:
        raise HTTPException(
            status_code=400,
            detail="rag_page_id 與該 rag_unit_id 所屬之 Rag_Unit.rag_page_id 不一致",
        )
    try:
        ut = int(row.get("unit_type") or 0)
    except (TypeError, ValueError):
        ut = 0
    if ut != RAG_UNIT_TYPE_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"僅 unit_type=2（文字單元）可使用此端點，目前 unit_type={ut}",
        )

    text_file_name = (row.get("text_file_name") or "").strip()
    transcript = transcript_from_row(row)

    zip_folder = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if not transcript and zip_folder:
        try:
            transcript, zip_text_name = _transcript_from_upload_zip_for_folder(
                owner_pid, tab, zip_folder
            )
            if not text_file_name:
                text_file_name = zip_text_name
        except (FileNotFoundError, ValueError) as e:
            _logger.debug("GET /rag/unit/text ZIP 備援略過: %s", e)
        except Exception:
            _logger.exception("GET /rag/unit/text ZIP 備援失敗")

    return RagUnitTextResponse(
        rag_page_id=tab,
        folder_name=zip_folder,
        rag_unit_id=unit_id,
        text_file_name=text_file_name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/unit/mp3-file
# ---------------------------------------------------------------------------


@router.get(
    "/unit/mp3-file",
    summary="Rag Unit Audio File",
    operation_id="rag_unit_mp3_file",
    response_model=RagUnitMp3FileFromZipResponse,
)
def rag_unit_audio_file(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同",
    ),
):
    """
    自 upload ZIP 內指定資料夾擷取音訊（base64）與**恰好一個**文字檔全文作為 `transcript`（與 build-rag-zip unit_type=3 一致；須音訊＋逐字稿）。
    query 須含 `person_id`，且須與該 `rag_page_id` 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        transcript, text_file_name = read_mp3_unit_transcript_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    media = audio_media_type_for_suffix(suffix)
    disp_name = Path(inner_path).name
    return RagUnitMp3FileFromZipResponse(
        rag_page_id=tab,
        folder_name=folder,
        audio_base64=base64.b64encode(contents).decode(),
        media_type=media,
        filename=disp_name,
        text_file_name=text_file_name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/unit/youtube-url
# ---------------------------------------------------------------------------


@router.get(
    "/unit/youtube-url",
    summary="Rag Unit Youtube Url",
    operation_id="rag_unit_youtube_url",
    response_model=RagUnitYoutubeUrlFromZipResponse,
)
def rag_unit_youtube_url(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = Query(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(
        ...,
        description="與 Rag_Unit.unit_name、upload ZIP 內單元資料夾名相同",
    ),
):
    """
    自 upload ZIP 內指定資料夾讀取**恰好一個**文字檔：第一行為 YouTube URL，第二行起為 `transcript`（與 build-rag-zip unit_type=4 一致）。
    query 須含 `person_id`，且須與該 rag_page_id 之 Rag.person_id 一致。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        vid, inner_path = read_youtube_video_id_from_upload_zip(zip_bytes, folder)
        transcript, _ = read_supplementary_text_from_youtube_unit(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RagUnitYoutubeUrlFromZipResponse(
        rag_page_id=tab,
        folder_name=folder,
        youtube_url=f"https://www.youtube.com/watch?v={vid}",
        text_file_name=Path(inner_path).name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm_api_key
# ---------------------------------------------------------------------------


class RagApiKeyResponse(BaseModel):
    """GET/PUT /rag/llm_api_key 回應（Course_Setting key=rag-api-key）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class PutRagApiKeyRequest(BaseModel):
    """PUT /rag/llm_api_key 的 body。"""

    api_key: str = Field(..., description="RAG LLM API Key")


@router.get("/llm_api_key", response_model=RagApiKeyResponse)
def get_rag_api_key_setting(person_id: PersonId, course_id: CourseId):
    """讀取 RAG LLM API Key（Course_Setting key=rag-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_api_key_setting_row(COURSE_SETTING_RAG_API_KEY, course_id)
    if not row:
        return RagApiKeyResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return RagApiKeyResponse(
        course_setting_id=row.get("course_setting_id"),
        course_id=course_id,
        api_key=value or None,
    )


@router.put("/llm_api_key", response_model=RagApiKeyResponse)
def put_rag_api_key_setting(
    body: openapi_body(PutRagApiKeyRequest, {"api_key": "sk-..."}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 RAG LLM API Key（Course_Setting key=rag-api-key，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.api_key or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            COURSE_SETTING_RAG_API_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            return RagApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return RagApiKeyResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=course_id,
            api_key=saved or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm_model
# ---------------------------------------------------------------------------


class RagLlmModelResponse(BaseModel):
    """GET/PUT /rag/llm_model 回應（Course_Setting key=llm-model；出題、批改、弱點分析共用）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutRagLlmModelRequest(BaseModel):
    """PUT /rag/llm_model 的 body。"""

    llm_model: str = Field(
        ...,
        description="RAG 出題／批改／弱點分析 LLM 模型名（對應 QUIZ_LLM_MODEL／GRADE_LLM_MODEL／WEAKNESS_LLM_MODEL，預設 gpt-5.4）",
    )


@router.get("/llm_model", response_model=RagLlmModelResponse)
def get_rag_llm_model_setting(person_id: PersonId, course_id: CourseId):
    """讀取 RAG 出題／批改／弱點分析 LLM 模型（Course_Setting key=llm-model，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    row = fetch_api_key_setting_row(COURSE_SETTING_LLM_MODEL, course_id)
    if not row:
        return RagLlmModelResponse(course_id=course_id)
    value = (row.get("value") or "").strip()
    return RagLlmModelResponse(
        course_setting_id=row.get("course_setting_id"),
        course_id=course_id,
        llm_model=value or None,
    )


@router.put("/llm_model", response_model=RagLlmModelResponse)
def put_rag_llm_model_setting(
    body: openapi_body(PutRagLlmModelRequest, {"llm_model": "gpt-5.4"}),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 RAG 出題／批改／弱點分析 LLM 模型（Course_Setting key=llm-model，依 course_id）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.llm_model or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            COURSE_SETTING_LLM_MODEL,
            value_to_save,
            course_id,
        )
        if not row:
            return RagLlmModelResponse(course_id=course_id, llm_model=value_to_save or None)
        saved = (row.get("value") or "").strip()
        return RagLlmModelResponse(
            course_setting_id=row.get("course_setting_id"),
            course_id=course_id,
            llm_model=saved or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

