"""
Exam API 模組。對應 public.Exam、public.Exam_Quiz。

檔案結構（由上而下）：模型／檢索說明 → LLM Prompt → 型別與作業快取 → Pydantic → 輔助函式與路由。

- GET /exam/tabs：列出 Exam（deleted=false，person_id 篩選，local 篩選，course_id 篩選），每筆帶 quizzes[]（Exam_Quiz，含 follow_up 鏈）。
- GET /exam/rag-for-exams：列出 for_exam 測驗用 RAG 資料（Rag_Unit.for_exam=true 或含 Rag_Quiz.for_exam=true）；須傳 course_id；僅含隸屬 Rag.local 與 query local 相符之分頁（未傳 local 時依連線是否本機判定）。
- POST /exam/tab/create：建立一筆 Exam（寫入 course_id）。
- PUT /exam/tab/tab-name：更新既有 Exam 的 tab_name。
- POST /exam/tab/quiz/create：新增空白 Exam_Quiz（不呼叫 LLM）。
- POST /exam/tab/quiz/create-llm-generate：新增 Exam_Quiz 並 LLM 出題（等同 create 後 llm-generate）。
- POST /exam/tab/quiz/llm-generate：出題須含 `rag_tab_id`／`rag_unit_id`／`rag_quiz_id`；成功後更新 Exam_Quiz（含 `unit_name`、`quiz_user_prompt_text`、`answer_user_prompt_text`）；JSON 回傳含兩段 prompt 供前端顯示。
- POST /exam/tab/quiz/create-llm-generate-followup：新增 Exam_Quiz 並接續 LLM 出題（等同 create 後 llm-generate-followup；第一題時自動一般出題）。
- POST /exam/tab/quiz/llm-generate-followup：接續出題；須請求提供 follow_up_exam_quiz_id（>0）與非空 quiz_history_list，否則視為第一題（回應不含 follow_up）；答不好追問弱點，答好則出新題。
- POST /exam/tab/quiz/llm-grade：非同步 RAG+LLM 評分；回傳 202 + job_id，輪詢 GET /exam/tab/quiz/grade-result/{job_id}。
- PUT /exam/tab/delete/{exam_tab_id}：軟刪除 Exam。
- POST /exam/tab/quiz/rate：更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。
"""

import json
import logging
import shutil
import tempfile
import textwrap
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse, Response
from postgrest.exceptions import APIError
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from services.quiz_generation import (
    generate_quiz,
    generate_quiz_followup,
    generate_quiz_followup_transcript_only,
    generate_quiz_transcript_only,
)
from utils.openapi import openapi_body

from services.exam_queries import (
    ensure_exam_quiz_rag_id_keys,
    enrich_exam_quizzes_rag_tab_from_units,
    exam_default_row,
    exams_table_select,
    exam_tab_quizzes_response,
    quizzes_by_exam_tab_ids,
    rag_quiz_for_exam_response_row,
    select_rag_row_with_transcript_fallback,
)
from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_exam_quiz_with_grade,
)
from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.retry import call_with_transient_http_retry
from utils.serialization import to_json_safe
from utils.llm_key import get_llm_api_key
from utils.exam_course import require_exam_row
from utils.rag_course import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.rag_exam_setting import is_localhost_request, rag_id_from_rag_tab_id, resolve_exam_content_rag_id
from utils.rag_stem import get_rag_stem_from_rag_id, instruction_from_rag_row, transcript_from_row
from utils.supabase import get_supabase
from utils.zip_storage import generate_tab_id, get_zip_path

router = APIRouter(prefix="/exam", tags=["exam"])

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模型與檢索常數
# ---------------------------------------------------------------------------
# 本模組不直接宣告 OpenAI 模型名；出題呼叫 `services.quiz_generation`（`QUIZ_LLM_MODEL`、embedding、k），
# 批改呼叫 `services.grading`（`GRADE_LLM_MODEL`、檢索與 chunk 常數）。


# ---------------------------------------------------------------------------
# LLM Prompt（exam 出題 user 前綴；置於課程內容／檢索片段之前）
# ---------------------------------------------------------------------------
# 由 _exam_llm_generate_api_instruction 組字；欄位名與順序同 public.Exam_Quiz（至 quiz_user_prompt_text）。

ExamQuizRateValue = Literal[-1, 0, 1]

_exam_grade_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ListExamResponse(BaseModel):
    """GET /exam/tabs 回應：每筆 Exam 含 quizzes[]（Exam_Quiz，含 follow_up 鏈）。"""
    exams: list[dict] = Field(
        ...,
        description="每筆 Exam 的 quizzes[] 為 Exam_Quiz（含 follow_up、follow_up_quiz 鏈、答案欄位）",
    )
    count: int


class ListRagForExamsResponse(BaseModel):
    """GET /exam/rag-for-exams：單元為完整 Rag_Unit；quizzes 含 RAG 關聯鍵與出題／批改 prompt。"""
    units: list[dict] = Field(
        ...,
        description="Rag_Unit 列；每筆 quizzes[] 含 follow_up、rag_quiz_id、rag_tab_id、rag_unit_id、person_id、quiz_name、quiz_user_prompt_text、quiz_content、quiz_hint、quiz_answer_reference、answer_user_prompt_text",
    )
    count: int


class CreateExamRequest(BaseModel):
    """POST /exam/tab/create：欄位順序同 public.Exam（exam_tab_id, person_id, tab_name, local；不含 exam_id／course_id／deleted／時間戳）。"""
    exam_tab_id: str | None = Field(None, description="選填；未傳則由後端產生")
    person_id: str = Field("", description="選填，寫入 Exam.person_id")
    tab_name: str = Field("", description="測驗顯示名稱")
    local: bool = Field(False, description="是否為本機 Exam")


class UpdateExamUnitNameRequest(BaseModel):
    """PUT /exam/tab/tab-name：以 exam_id（主鍵）更新 tab_name。"""
    exam_id: int = Field(..., description="Exam 主鍵")
    tab_name: str = Field(..., description="新的顯示名稱")


class ExamCreateQuizRequest(BaseModel):
    """POST /exam/tab/quiz/create：新增空白 Exam_Quiz（無 LLM）。僅 exam_tab_id（不傳 rag_unit_id）。"""
    exam_tab_id: str = Field("", description="目標 Exam 的 exam_tab_id")


class ExamCreateLlmGenerateQuizRequest(BaseModel):
    """POST /exam/tab/quiz/create-llm-generate；先 create 再 llm-generate，不需傳 exam_quiz_id。"""

    exam_tab_id: str = Field("", description="目標 Exam 的 exam_tab_id")
    rag_tab_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_tab_id（與 POST /rag/tab/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    quiz_history_list: list[str] = Field(
        default_factory=list,
        description="已出過的題目題幹（字串陣列）；送入 LLM 出題 prompt，避免重複出題",
    )


class ExamLlmGenerateQuizRequest(BaseModel):
    """POST /exam/tab/quiz/llm-generate；請求 body 欄位順序同 public.Exam_Quiz（exam_quiz_id, rag_tab_id, rag_unit_id, rag_quiz_id），末欄 quiz_history_list 為 API 擴充以避免重複出題。"""

    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵")
    rag_tab_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_tab_id（與 POST /rag/tab/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz。列鎖鍵規則同 rag_unit_id",
    )
    quiz_history_list: list[str] = Field(
        default_factory=list,
        description="已出過的題目題幹（字串陣列）；送入 LLM 出題 prompt，避免重複出題",
    )


class ExamQuizHistoryPair(BaseModel):
    """先前問答一組：題幹、作答、參考答案與評閱（對齊 Exam_Quiz 欄位）。"""

    quiz_content: str = Field(..., description="先前題目題幹")
    answer_content: str = Field(
        ...,
        description="先前作答（學生答案）",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    quiz_answer_reference: str = Field(
        "",
        description="該題參考答案（對齊 Exam_Quiz.quiz_answer_reference）",
        validation_alias=AliasChoices(
            "quiz_answer_reference",
            "quiz_reference_answer",
            "reference_answer",
        ),
    )
    answer_critique: str = Field(
        "",
        description="該題評閱／批改評語（對齊 Exam_Quiz.answer_critique）",
        validation_alias=AliasChoices("answer_critique", "critique", "quiz_comments"),
    )


class ExamCreateLlmGenerateQuizFollowupRequest(BaseModel):
    """POST /exam/tab/quiz/create-llm-generate-followup；先 create 再 llm-generate-followup，不需傳 exam_quiz_id。"""

    model_config = ConfigDict(populate_by_name=True)

    exam_tab_id: str = Field("", description="目標 Exam 的 exam_tab_id")
    rag_tab_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_tab_id（與 POST /rag/tab/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    follow_up_exam_quiz_id: int = Field(
        0,
        ge=0,
        description="前一筆 Exam_Quiz 主鍵；>0 時寫入本列 follow_up=true 與此 id。傳 0 視為第一題（一般出題）",
        validation_alias=AliasChoices("follow_up_exam_quiz_id", "followUpExamQuizId"),
    )
    quiz_history_list: list[ExamQuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答；非空時才用追問 LLM prompt。每項含 quiz_content、answer_content、quiz_answer_reference、answer_critique",
        validation_alias=AliasChoices("quiz_history_list", "quizHistoryList"),
    )


class ExamLlmGenerateQuizFollowupRequest(BaseModel):
    """POST /exam/tab/quiz/llm-generate-followup；欄位順序同 Exam_Quiz 至 rag_quiz_id，接 follow_up_exam_quiz_id 與 quiz_history_list。"""

    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（本筆接續題）")
    rag_tab_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_tab_id（與 POST /rag/tab/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    follow_up_exam_quiz_id: int = Field(
        0,
        ge=0,
        description="前一筆 Exam_Quiz 主鍵；>0 時寫入本列 follow_up=true 與此 id。傳 0 視為第一題（一般出題）",
        validation_alias=AliasChoices("follow_up_exam_quiz_id", "followUpExamQuizId"),
    )
    quiz_history_list: list[ExamQuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答；非空時才用追問 LLM prompt。每項含 quiz_content、answer_content、quiz_answer_reference、answer_critique",
        validation_alias=AliasChoices("quiz_history_list", "quizHistoryList"),
    )


class ExamQuizRateRequest(BaseModel):
    """POST /exam/tab/quiz/rate：更新 Exam_Quiz.quiz_rate。"""
    exam_quiz_id: int = Field(..., ge=1, description="Exam_Quiz 主鍵")
    quiz_rate: ExamQuizRateValue = Field(0, description="僅 -1、0、1")


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/tab/quiz/llm-grade：body 欄位依序對應 public.Exam_Quiz 之 exam_quiz_id、quiz_content、answer_content（學生作答 quiz_answer）。
    批改用 quiz_user_prompt_text／answer_user_prompt_text 優先採 Exam_Quiz 列，缺則自 Rag_Quiz 補齊。"""
    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（必填，>0）；置入評分 prompt")
    quiz_content: str = Field(
        "",
        description="選填；若空則使用 Exam_Quiz 中存的 quiz_content；置入評分 prompt",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答；寫入 Exam_Quiz.answer_content；置入評分 prompt",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


# ---------------------------------------------------------------------------
# 路由內輔助（僅限此模組）
# ---------------------------------------------------------------------------

def _safe_unlink(p: Path) -> None:
    """刪除暫存檔；忽略檔案不存在或刪除失敗。"""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _load_exam_for_quiz(
    supabase: Any,
    *,
    exam_id: int,
    exam_tab_id: str,
    caller_person_id: str,
    course_id: int,
) -> tuple[str, str]:
    """回傳 (exam_tab_id, person_id)。需擇一傳入 exam_id 或 exam_tab_id；須符合 course_id。"""
    _ = supabase
    row = require_exam_row(
        course_id=course_id,
        exam_id=exam_id,
        exam_tab_id=exam_tab_id,
        person_id=caller_person_id,
    )
    out_tab = (row.get("exam_tab_id") or "").strip()
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該 Exam 的 person_id 為空")
    return out_tab, person_id


def _exam_llm_generate_api_instruction(
    *,
    exam_quiz_id: int,
    exam_tab_id: str | None,
    rag_tab_id: str | None,
    rag_unit_id: int,
    rag_quiz_id: int | None,
    person_id: str | None,
    unit_name: str | None,
    quiz_name: str | None,
    quiz_user_prompt_text: str,
) -> str:
    """
    組出 POST /exam/tab/quiz/llm-generate 送進 utils.generate_quiz* 的 quiz_user_prompt_text 前綴。
    參數名與順序同 public.Exam_Quiz（至 quiz_user_prompt_text）；rag_quiz_id 列於提示時可為未關聯說明字串。
    """
    try:
        rqi = int(rag_quiz_id) if rag_quiz_id is not None else 0
    except (TypeError, ValueError):
        rqi = 0
    rag_quiz_id_line = f"`{rqi}`" if rqi > 0 else "（Exam_Quiz 未關聯 rag_quiz_id）"
    exam_tab_id_s = (exam_tab_id or "").strip() or "（未提供）"
    rag_tab_id_s = (rag_tab_id or "").strip() or "（未提供）"
    person_id_s = (person_id or "").strip() or "（未提供）"
    unit_name_s = (unit_name or "").strip() or "（未提供）"
    quiz_name_s = (quiz_name or "").strip() or "（未提供）"
    quiz_user_prompt_text_s = (quiz_user_prompt_text or "").strip() or "（未提供）"
    ru = int(rag_unit_id or 0)
    return textwrap.dedent(f"""
        ## 本次請求 API 參數

        請一併納入出題考量（欄位順序同 public.Exam_Quiz：exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, unit_name, quiz_name, quiz_user_prompt_text, …）。

        - **exam_quiz_id**：`{exam_quiz_id}`
        - **exam_tab_id**：{exam_tab_id_s}
        - **rag_tab_id**：{rag_tab_id_s}
        - **rag_unit_id**：`{ru}`
        - **rag_quiz_id**：{rag_quiz_id_line}
        - **person_id**：{person_id_s}
        - **unit_name**：{unit_name_s}
        - **quiz_name**：{quiz_name_s}

        ### quiz_user_prompt_text

        {quiz_user_prompt_text_s}
        """).strip()


# ---------------------------------------------------------------------------
# GET /exam/tabs
# ---------------------------------------------------------------------------

@router.get("/tabs", response_model=ListExamResponse)
def list_exams(
    request: Request,
    person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 Exam.local 與此值相同的列。未傳時：本機連線視為 true，否則 false",
    ),
):
    """列出 Exam（deleted=false，person_id 篩選，local 篩選）。每筆 Exam 帶 quizzes[]（Exam_Quiz，含 follow_up 鏈）。"""

    def _list_exams_once() -> ListExamResponse:
        local_filter = local if local is not None else is_localhost_request(request)
        data = exams_table_select(exclude_deleted=True, local_match=local_filter, course_id=course_id)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        tab_ids = list(dict.fromkeys(
            str(r.get("exam_tab_id")) for r in data if r.get("exam_tab_id") is not None
        ))
        quizzes_by_tab = quizzes_by_exam_tab_ids(tab_ids, course_id=course_id)
        flat_qz = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_qz)
        ensure_exam_quiz_rag_id_keys(flat_qz)

        for row in data:
            tab_id = str(row.get("exam_tab_id") or "")
            row["quizzes"] = exam_tab_quizzes_response(quizzes_by_tab.get(tab_id, []))

        data = to_json_safe(data)
        return ListExamResponse(exams=data, count=len(data))

    try:
        return call_with_transient_http_retry(_list_exams_once)
    except Exception as e:
        _logger.exception("GET /exam/tabs 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Exam 失敗: {e!s}")


# ---------------------------------------------------------------------------
# GET /exam/rag-for-exams
# ---------------------------------------------------------------------------

@router.get(
    "/rag-for-exams",
    response_model=ListRagForExamsResponse,
    summary="List RAG units & quizzes marked for exam",
)
def list_rag_for_exams(
    request: Request,
    _person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 rag_tab_id 隸屬 Rag.local 與此值相同之單元／題目。未傳時：本機連線視為 true，否則 false",
    ),
):
    """
    回傳 for-exam 相關 RAG 單元與題目（不限 person_id）：
    - 僅 rag_tab_id 對應之 Rag 列（deleted=false）其 local 與 query local 相符者（未傳 local 時同 GET /exam/tabs 依連線判定）。
    - 單元：Rag_Unit.deleted=false 且（Rag_Unit.for_exam=true 或至少一筆 Rag_Quiz.for_exam=true 隸屬該 rag_unit_id）。
    - quizzes：僅 Rag_Quiz.for_exam=true 且 deleted=false。
    """

    def _list_rag_for_exams_once() -> ListRagForExamsResponse:
        supabase = get_supabase()
        local_filter = local if local is not None else is_localhost_request(request)

        tabs_for_local = (
            supabase.table("Rag")
            .select("rag_tab_id")
            .eq("deleted", False)
            .eq("local", local_filter)
            .eq("course_id", course_id)
            .execute()
            .data
            or []
        )
        allowed_tab_ids = list(dict.fromkeys(
            r["rag_tab_id"] for r in tabs_for_local if r.get("rag_tab_id") is not None
        ))
        if not allowed_tab_ids:
            return ListRagForExamsResponse(units=[], count=0)

        def build_exam_quizzes(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id, follow_up, quiz_name, quiz_user_prompt_text, "
                "quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("for_exam", True)
                .eq("deleted", False)
                .in_("rag_tab_id", allowed_tab_ids)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        quizzes_resp = execute_with_course_id_fallback("Rag_Quiz", build_exam_quizzes, course_id)
        quiz_rows = quizzes_resp.data or []

        unit_ids_from_quizzes: set[int] = set()
        quizzes_by_unit: dict[int, list[dict]] = {}
        for q in quiz_rows:
            q_uid = q.get("rag_unit_id")
            if q_uid is None:
                continue
            try:
                uid = int(q_uid)
            except (TypeError, ValueError):
                continue
            unit_ids_from_quizzes.add(uid)
            quizzes_by_unit.setdefault(uid, []).append(rag_quiz_for_exam_response_row(q))

        unit_ids_from_units: set[int] = set()
        try:
            def build_units_flag(with_course_filter: bool):
                q = (
                    supabase.table("Rag_Unit")
                    .select("rag_unit_id")
                    .eq("for_exam", True)
                    .eq("deleted", False)
                    .in_("rag_tab_id", allowed_tab_ids)
                )
                if with_course_filter and course_id is not None:
                    q = q.eq("course_id", course_id)
                return q

            units_flag_resp = execute_with_course_id_fallback(
                "Rag_Unit", build_units_flag, course_id
            )
            for u in units_flag_resp.data or []:
                rid = u.get("rag_unit_id")
                if rid is not None:
                    try:
                        unit_ids_from_units.add(int(rid))
                    except (TypeError, ValueError):
                        pass
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and "for_exam" in msg:
                _logger.warning("Rag_Unit 無 for_exam 欄位；GET /exam/rag-for-exams 僅列出含 Rag_Quiz.for_exam 之單元")
            else:
                raise

        all_unit_ids = list(dict.fromkeys(list(unit_ids_from_units | unit_ids_from_quizzes)))
        if not all_unit_ids:
            return ListRagForExamsResponse(units=[], count=0)

        def build_exam_units(with_course_filter: bool):
            q = (
                supabase.table("Rag_Unit")
                .select("*")
                .in_("rag_unit_id", all_unit_ids)
                .in_("rag_tab_id", allowed_tab_ids)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        units = (
            execute_with_course_id_fallback("Rag_Unit", build_exam_units, course_id).data or []
        )
        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        out = to_json_safe(units)
        return ListRagForExamsResponse(units=out, count=len(out))

    try:
        return call_with_transient_http_retry(_list_rag_for_exams_once)
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("GET /exam/rag-for-exams 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 for_exam RAG 失敗: {e!s}")


# ---------------------------------------------------------------------------
# POST /exam/tab/create
# ---------------------------------------------------------------------------

@router.post("/tab/create")
def create_exam(
    body: openapi_body(
        CreateExamRequest,
        {"exam_tab_id": "", "person_id": "", "tab_name": "", "local": False},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """建立一筆 Exam。exam_tab_id 可選（未傳由後端產生）；local 選填（預設 false）。"""
    fid = (body.exam_tab_id or "").strip()
    body_pid = (body.person_id or "").strip()
    person_id = body_pid if body_pid else caller_person_id
    if body_pid and body_pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    if not fid:
        fid = generate_tab_id(person_id or None)
    if "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")
    tab_name = (body.tab_name or "").strip()

    supabase = get_supabase()
    ins = (
        supabase.table("Exam")
        .insert(exam_default_row(fid, tab_name=tab_name, person_id=person_id, course_id=course_id, local=body.local))
        .execute()
    )
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="建立 Exam 失敗")
    row = ins.data[0]
    return {
        "exam_id": row.get("exam_id"),
        "exam_tab_id": row.get("exam_tab_id", fid),
        "tab_name": row.get("tab_name", tab_name),
        "person_id": row.get("person_id", person_id),
        "course_id": row.get("course_id", course_id),
        "local": row.get("local", body.local),
        "deleted": row.get("deleted", False),
        "updated_at": to_taipei_iso(row.get("updated_at")),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


# ---------------------------------------------------------------------------
# PUT /exam/tab/tab-name
# ---------------------------------------------------------------------------

@router.put("/tab/tab-name", summary="Update Exam Tab Name")
def update_exam_unit_tab_name(
    body: openapi_body(UpdateExamUnitNameRequest, {"exam_id": 1, "tab_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """更新既有 Exam 的 tab_name（以 exam_id 定位；僅 deleted=false）。"""
    if body.exam_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 exam_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Exam")
            .select("exam_id, exam_tab_id, tab_name, person_id, course_id, local, deleted")
            .eq("exam_id", body.exam_id)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 exam_id 的 Exam 資料，或已刪除")
        row = sel.data[0]
        fid = row.get("exam_tab_id")
        pid = row.get("person_id")
        if (pid or "").strip() != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Exam")
        ts = now_taipei_iso()
        supabase.table("Exam").update({"tab_name": tab_name, "updated_at": ts}).eq("exam_id", body.exam_id).eq("deleted", False).execute()
        return {
            "exam_id": body.exam_id,
            "exam_tab_id": fid,
            "tab_name": tab_name,
            "person_id": pid,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# PUT /exam/tab/delete/{exam_tab_id}
# ---------------------------------------------------------------------------

@router.put("/tab/delete/{exam_tab_id}", status_code=200, summary="Delete Exam Tab", operation_id="exam_tab_delete")
def delete_exam(
    caller_person_id: PersonId,
    course_id: CourseId,
    exam_tab_id: str = PathParam(..., description="要刪除的 Exam 的 exam_tab_id"),
):
    """PUT /exam/tab/delete/{exam_tab_id}。軟刪除：將 Exam 的 deleted 設為 true。"""
    fid = (exam_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")
    supabase = get_supabase()
    r = (
        supabase.table("Exam")
        .select("exam_id, person_id")
        .eq("exam_tab_id", fid)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="找不到該 exam_tab_id 的 Exam 資料，或已刪除")
    pid = (r.data[0].get("person_id") or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權刪除該 Exam")
    supabase.table("Exam").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("exam_tab_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
    return {"message": "已將 Exam 標記為刪除", "exam_tab_id": fid, "person_id": pid}


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/create
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/create", summary="Exam Create Quiz (no LLM)", operation_id="exam_create_quiz")
def exam_insert_empty_quiz(
    body: openapi_body(ExamCreateQuizRequest, {"exam_tab_id": "string"}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """新增一筆空白 Exam_Quiz，不呼叫 LLM；body 僅需 exam_tab_id。亦可改用 POST /exam/tab/quiz/create-llm-generate 一次完成建立與出題。"""
    try:
        return _create_exam_quiz_record(
            exam_tab_id=body.exam_tab_id,
            caller_person_id=caller_person_id,
            course_id=course_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /exam/tab/quiz/create 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/llm-generate
# ---------------------------------------------------------------------------

_EXAM_LLM_GEN_DESCRIPTION = """\
Body：`exam_quiz_id`、`rag_tab_id`、`rag_unit_id`、`rag_quiz_id` 皆必填（順序同 public.Exam_Quiz）。
`rag_tab_id` 須對應 `public.Rag.rag_tab_id`，且與所列 `rag_unit_id`、`rag_quiz_id` 在 DB 上所隸屬之 Tab 一致；並用此載入 ZIP／單元（**不依賴** System_Setting 之 `rag_localhost`/`rag_deploy`）。
若該 Exam_Quiz 列**已有**有效的 `rag_unit_id`、`rag_quiz_id`，請求兩鍵須與列**完全一致**，否則 400。
若列**尚未**寫入（缺其一或為 0），則以此請求綁定，出題成功後一併寫回。
`quiz_user_prompt_text`／`answer_user_prompt_text` 僅自 Rag_Quiz（請求中的 `rag_quiz_id`）讀取，不另由 body 帶入文字；出題成功後寫入 Exam_Quiz 以記錄當下模板。
unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcript。
unit_type 2／3／4 時不載入 RAG ZIP，改以 transcript 純 LLM 出題。
選填 `quiz_history_list`（字串陣列）：已出過的題目題幹，由 `services.quiz_generation` 併入 user「已出過題目」區塊，避免重複出題。
出題成功後更新該筆 Exam_Quiz（`rag_tab_id`、`unit_name`（與 RAG 單元顯示名一致，供 GET /exam/tabs 分群）、`quiz_name`、quiz_content／quiz_hint／quiz_answer_reference、rag_unit_id、rag_quiz_id；自該 `rag_quiz_id` 之 Rag_Quiz 寫入 `quiz_user_prompt_text`、`answer_user_prompt_text` 以記錄當下模板；清空作答欄位）。

**回應 JSON**（除題目欄位外）必含：`quiz_user_prompt_text`、`answer_user_prompt_text`（與寫入 Exam_Quiz 之快照相同，供前端顯示出題／作答模板）；`unit_name` 與資料庫更新後一致。
"""

_EXAM_LLM_GENERATE_OPENAPI_EXAMPLES = {
    "exam_quiz_id": 1,
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "quiz_history_list": ["先前已出過的題幹文字"],
}

_EXAM_CREATE_LLM_GENERATE_OPENAPI_EXAMPLES = {
    "exam_tab_id": "string",
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "quiz_history_list": ["先前已出過的題幹文字"],
}

_EXAM_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES = {
    "exam_quiz_id": 2,
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "follow_up_exam_quiz_id": 1,
    "quiz_history_list": [
        {
            "quiz_content": "先前題目題幹",
            "answer_content": "學生先前作答",
            "quiz_answer_reference": "參考答案全文",
            "answer_critique": "批改評語（指出答不好之處）",
        },
    ],
}


_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES = {
    "exam_tab_id": "string",
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "follow_up_exam_quiz_id": 1,
    "quiz_history_list": [
        {
            "quiz_content": "先前題目題幹",
            "answer_content": "學生先前作答",
            "quiz_answer_reference": "參考答案全文",
            "answer_critique": "批改評語（指出答不好之處）",
        },
    ],
}


def _create_exam_quiz_record(
    *,
    exam_tab_id: str,
    caller_person_id: str,
    course_id: int,
) -> dict[str, Any]:
    """新增空白 Exam_Quiz；回傳 enrich 後列。"""
    supabase = get_supabase()
    resolved_tab_id, person_id = _load_exam_for_quiz(
        supabase,
        exam_id=0,
        exam_tab_id=exam_tab_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )
    qts = now_taipei_iso()
    quiz_row: dict[str, Any] = {
        "exam_tab_id": resolved_tab_id,
        "rag_tab_id": "",
        "rag_unit_id": None,
        "rag_quiz_id": None,
        "person_id": person_id,
        "course_id": course_id,
        "follow_up": False,
        "follow_up_exam_quiz_id": 0,
        "unit_name": "",
        "quiz_name": "",
        "quiz_user_prompt_text": None,
        "quiz_content": "",
        "quiz_hint": "",
        "quiz_answer_reference": "",
        "quiz_rate": 0,
        "answer_user_prompt_text": None,
        "answer_content": None,
        "answer_critique": None,
        "created_at": qts,
        "updated_at": qts,
    }
    ins = supabase.table("Exam_Quiz").insert(quiz_row).execute()
    if not ins.data or len(ins.data) == 0:
        raise HTTPException(status_code=500, detail="寫入 Exam_Quiz 失敗（無回傳資料）")
    row = dict(ins.data[0])
    enrich_exam_quizzes_rag_tab_from_units([row])
    ensure_exam_quiz_rag_id_keys([row])
    return to_json_safe(row)


def _exam_quiz_history_qa_dicts(pairs: list[ExamQuizHistoryPair]) -> list[dict[str, str]]:
    return [
        {
            "quiz_content": p.quiz_content,
            "answer_content": p.answer_content,
            "quiz_answer_reference": p.quiz_answer_reference,
            "answer_critique": p.answer_critique,
        }
        for p in pairs
    ]


def _resolve_exam_followup_mode(
    *,
    followup_requested: bool,
    follow_up_exam_quiz_id: int,
    exam_quiz_id: int,
    quiz_history_qa: list[ExamQuizHistoryPair] | None,
) -> tuple[bool, bool, int, list[str], list[dict[str, str]]]:
    """
    follow_up_exam_quiz_id 以請求傳入為準。
    回傳 (use_followup_llm, mark_follow_up, follow_up_exam_quiz_id, history_stems, qa_dicts)。

    mark_follow_up：followup 端點且 follow_up_exam_quiz_id>0 → 寫入 follow_up=true。
    use_followup_llm：mark_follow_up 且 quiz_history_list 非空 → 使用追問 LLM prompt。
    """
    request_qa = _exam_quiz_history_qa_dicts(quiz_history_qa or [])

    def _history_stems(qa: list[dict[str, str]]) -> list[str]:
        return [
            (d.get("quiz_content") or "").strip()
            for d in qa
            if (d.get("quiz_content") or "").strip()
        ]

    if not followup_requested:
        return False, False, 0, _history_stems(request_qa), request_qa

    resolved_id = int(follow_up_exam_quiz_id or 0)
    if resolved_id <= 0 or resolved_id == exam_quiz_id:
        return False, False, 0, _history_stems(request_qa), request_qa

    mark_follow_up = True
    use_followup_llm = bool(request_qa)
    return use_followup_llm, mark_follow_up, resolved_id, _history_stems(request_qa), request_qa


def _select_rag_unit_for_exam_prompt(
    supabase: Any,
    *,
    rag_tab_id_for_units: str,
    course_id: int,
    stem_rag_unit_id: int | None,
    unit_filter: str | None,
) -> dict | None:
    """依 rag_tab_id 列出 Rag_Unit（含欄位降級相容），挑出對應 stem_rag_unit_id 或 unit_filter 的單元；無 rag_tab_id 時回傳 None。"""
    selected: dict | None = None
    if rag_tab_id_for_units:
        def _unit_q_select(cols: str, with_course_filter: bool):
            c = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(c)
                .eq("rag_tab_id", rag_tab_id_for_units)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        def _unit_q_execute(cols: str):
            return execute_with_course_id_fallback(
                "Rag_Unit",
                lambda wc: _unit_q_select(cols, wc),
                course_id,
            )

        _cols_full = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _cols_no_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _cols_no_fc = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _cols_min = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _cols_legacy_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        _cols_no_fc_legacy_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        try:
            unit_q = _unit_q_execute(_cols_full)
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code != "42703":
                raise
            if "transcript" in msg:
                try:
                    unit_q = _unit_q_execute(_cols_legacy_tr)
                except APIError as e_legacy:
                    if e_legacy.code == "42703" and "transcription" in (e_legacy.message or "").lower():
                        try:
                            unit_q = _unit_q_execute(_cols_no_tr)
                        except APIError as e2:
                            if e2.code == "42703" and "folder_combination" in (e2.message or "").lower():
                                unit_q = _unit_q_execute(_cols_min)
                            else:
                                raise
                    else:
                        raise
            elif "folder_combination" in msg:
                try:
                    unit_q = _unit_q_execute(_cols_no_fc)
                except APIError as e2:
                    if e2.code == "42703" and "transcript" in (e2.message or "").lower():
                        try:
                            unit_q = _unit_q_execute(_cols_no_fc_legacy_tr)
                        except APIError as e3:
                            if e3.code == "42703" and "transcription" in (e3.message or "").lower():
                                unit_q = _unit_q_execute(_cols_min)
                            else:
                                raise
                    else:
                        raise
            else:
                raise
        units = unit_q.data or []
        if stem_rag_unit_id and stem_rag_unit_id > 0:
            for u in units:
                try:
                    if int(u.get("rag_unit_id") or 0) == stem_rag_unit_id:
                        selected = u
                        break
                except (TypeError, ValueError):
                    continue
        if selected is None and not unit_filter:
            selected = units[0] if units else None
        elif selected is None and unit_filter:
            for u in units:
                un = (u.get("unit_name") or "").strip()
                fc = (u.get("folder_combination") or "").strip()
                if un == unit_filter or fc == unit_filter:
                    selected = u
                    break
    return selected


def _exam_llm_generate_quiz_impl(
    *,
    exam_quiz_id: int,
    rag_tab_id: str,
    rag_unit_id: int,
    rag_quiz_id: int,
    caller_person_id: str,
    course_id: int,
    followup: bool,
    quiz_history_stems: list[str] | None = None,
    quiz_history_qa: list[ExamQuizHistoryPair] | None = None,
    follow_up_exam_quiz_id: int = 0,
    always_mark_follow_up: bool = False,
):
    supabase = get_supabase()
    qsel = (
        supabase.table("Exam_Quiz")
        .select(
            "exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
            "unit_name, quiz_name, created_at"
        )
        .eq("exam_quiz_id", exam_quiz_id)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if not qsel.data or len(qsel.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={exam_quiz_id} 的 Exam_Quiz")
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Exam_Quiz 出題")

    use_followup_llm, mark_follow_up, resolved_follow_up_id, history_stems, qa_dicts = (
        _resolve_exam_followup_mode(
            followup_requested=followup,
            follow_up_exam_quiz_id=follow_up_exam_quiz_id,
            exam_quiz_id=exam_quiz_id,
            quiz_history_qa=quiz_history_qa,
        )
    )
    if always_mark_follow_up:
        mark_follow_up = True
        resolved_follow_up_id = int(follow_up_exam_quiz_id or 0)
        if resolved_follow_up_id == exam_quiz_id:
            resolved_follow_up_id = 0
        use_followup_llm = bool(qa_dicts)

    row_ruid = 0
    rag_unit_val = qrow.get("rag_unit_id")
    if rag_unit_val is not None:
        try:
            row_ruid = int(rag_unit_val)
        except (TypeError, ValueError):
            row_ruid = 0

    row_rqid = 0
    legacy_rq = qrow.get("rag_quiz_id")
    if legacy_rq is not None:
        try:
            row_rqid = int(legacy_rq)
        except (TypeError, ValueError):
            row_rqid = 0

    body_ruid = int(rag_unit_id)
    body_rqid = int(rag_quiz_id)

    row_has_rag_pair = row_ruid > 0 and row_rqid > 0
    if row_has_rag_pair and (body_ruid != row_ruid or body_rqid != row_rqid):
        raise HTTPException(
            status_code=400,
            detail=(
                "請求之 rag_unit_id、rag_quiz_id 須與該筆 Exam_Quiz 列已存值完全一致；"
                f"列上為 rag_unit_id={row_ruid}、rag_quiz_id={row_rqid}"
            ),
        )

    effective_ruid = body_ruid
    effective_rqid = body_rqid

    cand_rag_qid = effective_rqid

    tab_strip = (rag_tab_id or "").strip()
    if not tab_strip:
        raise HTTPException(status_code=400, detail="rag_tab_id 不可為空白")

    row_rtab = ""
    _row_rt = qrow.get("rag_tab_id")
    if _row_rt is not None:
        row_rtab = str(_row_rt).strip()
    if row_rtab and row_rtab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_tab_id 須與該 Exam_Quiz 列已存 rag_tab_id 一致；"
                f"請求為 {tab_strip!r}，列上為 {row_rtab!r}"
            ),
        )

    def fetch_ru_one(*, include_folder_combination: bool):
        def build(with_course_filter: bool):
            base_cols = (
                "rag_tab_id, unit_name, folder_combination, course_id"
                if include_folder_combination
                else "rag_tab_id, unit_name, course_id"
            )
            cols = select_without_course_id_if_needed("Rag_Unit", base_cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", effective_ruid)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        return execute_with_course_id_fallback("Rag_Unit", build, course_id)

    try:
        ru_one = fetch_ru_one(include_folder_combination=True)
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            ru_one = fetch_ru_one(include_folder_combination=False)
        else:
            raise
    if not ru_one.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_unit_id={effective_ruid} 之 Rag_Unit")
    ru_tab = (ru_one.data[0].get("rag_tab_id") or "").strip()
    if ru_tab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_tab_id 須與 rag_unit_id 所隸 Rag Tab 一致；"
                f"請求為 {tab_strip!r}，Rag_Unit 為 {ru_tab!r}"
            ),
        )

    def build_rq_one(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Quiz",
            "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id, quiz_name, quiz_user_prompt_text, "
            "quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
            with_course_filter,
        )
        q = (
            supabase.table("Rag_Quiz")
            .select(cols)
            .eq("rag_quiz_id", effective_rqid)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    rq_one = execute_with_course_id_fallback("Rag_Quiz", build_rq_one, course_id)
    if not rq_one.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={effective_rqid} 之 Rag_Quiz")
    rq_row0 = rq_one.data[0]
    rq_tab = (rq_row0.get("rag_tab_id") or "").strip()
    if rq_tab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_tab_id 須與 rag_quiz_id 所隸 Rag Tab 一致；"
                f"請求為 {tab_strip!r}，Rag_Quiz 為 {rq_tab!r}"
            ),
        )

    rag_id_resolved = rag_id_from_rag_tab_id(supabase, tab_strip, course_id)
    if rag_id_resolved is None or rag_id_resolved <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 rag_tab_id={tab_strip!r} 對應之 Rag（deleted=false）",
        )

    quiz_user_prompt_resolved = (rq_row0.get("quiz_user_prompt_text") or "").strip()
    answer_user_prompt_resolved = (rq_row0.get("answer_user_prompt_text") or "").strip()

    _ru0 = ru_one.data[0]
    _ru_display = (_ru0.get("unit_name") or "").strip()
    _ru_folder = (_ru0.get("folder_combination") or "").strip()
    unit_filter: str | None = (_ru_folder or _ru_display or "").strip() or None
    stem_rag_unit_id: int | None = effective_ruid if effective_ruid > 0 else None
    if not unit_filter:
        unit_filter = (qrow.get("unit_name") or "").strip() or None

    api_key = get_llm_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 LLM API Key：環境變數 LLM_API_KEY 或 OPENAI_API_KEY（本機可寫入 .env）",
        )

    rag_rows = select_rag_row_with_transcript_fallback(supabase, rag_id_resolved)
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id_resolved} 的 Rag 資料，或已刪除")
    rag_row = rag_rows.data[0]
    assert_row_course_id(rag_row, course_id, "Rag")
    rag_id = int(rag_row.get("rag_id") or 0)
    rag_tab_id_for_units = (rag_row.get("rag_tab_id") or "").strip()

    stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
        supabase, rag_id, unit_name=unit_filter, rag_unit_id=stem_rag_unit_id
    )

    selected = _select_rag_unit_for_exam_prompt(
        supabase,
        rag_tab_id_for_units=rag_tab_id_for_units,
        course_id=course_id,
        stem_rag_unit_id=stem_rag_unit_id,
        unit_filter=unit_filter,
    )

    transcript_text = ""
    if selected:
        transcript_text = transcript_from_row(selected)
    if not transcript_text:
        transcript_text = instruction_from_rag_row(rag_row)

    try:
        unit_type_val = int(selected.get("unit_type") or 0) if selected else 0
    except (TypeError, ValueError):
        unit_type_val = 0

    if unit_type_val in (2, 3, 4) and not transcript_text:
        raise HTTPException(
            status_code=400,
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 設定 transcript，或經 POST /rag/tab/build-rag-zip 寫入",
        )

    prompt_rag_unit_id = int(selected.get("rag_unit_id") or 0) if selected else int(qrow.get("rag_unit_id") or 0)
    prompt_rag_qid = cand_rag_qid
    un_for_prompt = (qrow.get("unit_name") or "").strip() or None
    rq_quiz_name = (rq_row0.get("quiz_name") or "").strip()
    qn_for_prompt = (qrow.get("quiz_name") or "").strip() or rq_quiz_name or None
    api_instr = _exam_llm_generate_api_instruction(
        exam_quiz_id=exam_quiz_id,
        exam_tab_id=(qrow.get("exam_tab_id") or "").strip() or None,
        rag_tab_id=tab_strip,
        rag_unit_id=prompt_rag_unit_id,
        rag_quiz_id=prompt_rag_qid,
        person_id=person_id,
        unit_name=un_for_prompt,
        quiz_name=qn_for_prompt,
        quiz_user_prompt_text=quiz_user_prompt_resolved,
    )

    path: Path | None = None
    stems_for_generate = history_stems if followup else (quiz_history_stems or [])
    try:
        if unit_type_val in (2, 3, 4):
            if use_followup_llm:
                result = generate_quiz_followup_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list=qa_dicts,
                )
            else:
                result = generate_quiz_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list=stems_for_generate,
                )
        else:
            path = get_zip_path(rag_zip_tab_id)
            if not path or not path.exists():
                raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（tab_id={rag_zip_tab_id}）")
            if use_followup_llm:
                result = generate_quiz_followup(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list=qa_dicts,
                )
            else:
                result = generate_quiz(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list=stems_for_generate,
                )
        result["transcript"] = "" if unit_type_val == 1 else transcript_text
        result["rag_output"] = {"rag_tab_id": stem, "unit_name": stem, "filename": f"{stem}.zip"}

        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_answer_reference") or "").strip()
        result["quiz_content"] = qc
        result["quiz_hint"] = qh
        result["quiz_answer_reference"] = qref
        result["exam_quiz_id"] = exam_quiz_id
        qts = now_taipei_iso()
        unit_name_for_display = (
            _ru_display or (qrow.get("unit_name") or "").strip() or unit_filter or stem or ""
        ).strip()
        quiz_name = (
            rq_quiz_name
            or (qrow.get("quiz_name") or "").strip()
            or unit_name_for_display
            or ((stem or "").strip())
            or ""
        )
        result["quiz_name"] = quiz_name
        result["quiz_user_prompt_text"] = quiz_user_prompt_resolved
        result["answer_user_prompt_text"] = answer_user_prompt_resolved
        result["unit_name"] = unit_name_for_display
        quiz_update: dict[str, Any] = {
            "rag_tab_id": tab_strip,
            "rag_unit_id": int(rag_unit_id),
            "rag_quiz_id": int(rag_quiz_id),
            "unit_name": unit_name_for_display,
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": quiz_user_prompt_resolved,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_user_prompt_text": answer_user_prompt_resolved,
            "answer_content": None,
            "answer_critique": None,
            "updated_at": qts,
        }
        if mark_follow_up:
            quiz_update["follow_up"] = True
            quiz_update["follow_up_exam_quiz_id"] = resolved_follow_up_id
            result["follow_up"] = True
            result["follow_up_exam_quiz_id"] = resolved_follow_up_id
            if qa_dicts:
                result["quiz_history_list"] = qa_dicts
        result["created_at"] = to_taipei_iso(qrow.get("created_at"))
        result["rag_tab_id"] = tab_strip
        result["rag_unit_id"] = int(rag_unit_id)
        result["rag_quiz_id"] = int(rag_quiz_id)
        log_path = (
            "/exam/tab/quiz/llm-generate-followup"
            if use_followup_llm
            else "/exam/tab/quiz/llm-generate"
        )
        try:
            supabase.table("Exam_Quiz").update(quiz_update).eq("exam_quiz_id", exam_quiz_id).execute()
        except Exception as e:
            _logger.exception("POST %s 寫入 Exam_Quiz 失敗", log_path)
            raise HTTPException(status_code=500, detail=f"寫入 Exam_Quiz 失敗: {e!s}") from e
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


@router.post(
    "/tab/quiz/llm-generate",
    summary="Rag LLM Generate Quiz",
    operation_id="exam_llm_generate_quiz",
    description=_EXAM_LLM_GEN_DESCRIPTION.strip(),
)
@router.post("/generate-quiz", include_in_schema=False)
def exam_llm_generate_quiz(
    request: Request,
    body: openapi_body(ExamLlmGenerateQuizRequest, _EXAM_LLM_GENERATE_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """實作與說明見模組常數 `_EXAM_LLM_GEN_DESCRIPTION`（OpenAPI operation description）。亦可改用 POST /exam/tab/quiz/create-llm-generate 一次完成建立與出題。"""
    _ = request
    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=body.exam_quiz_id,
        rag_tab_id=body.rag_tab_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history_stems=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/llm-generate-followup
# ---------------------------------------------------------------------------

_EXAM_LLM_GEN_FOLLOWUP_DESCRIPTION = """\
Body：`exam_quiz_id`、`rag_tab_id`、`rag_unit_id`、`rag_quiz_id` 必填。
**追問鏈結**：請求 `follow_up_exam_quiz_id`（>0）時，出題成功後寫入本列 `follow_up=true` 與該 id（原樣）。
`quiz_history_list` 非空時才使用追問 LLM prompt；否則仍寫入 follow_up 但出題邏輯同一般 `llm-generate`。
`follow_up_exam_quiz_id` 為 0 或未傳則視為第一題，**回應不含** `follow_up`／`follow_up_exam_quiz_id`。
回應可含 `quiz_history_list` 與 `created_at`。
其餘 RAG 綁定、unit_type 出題邏輯同 `POST /exam/tab/quiz/llm-generate`。
`quiz_history_list` 為物件陣列，每項含 `quiz_content`、`answer_content`、`quiz_answer_reference`、`answer_critique`（一問一答一項）；
使用 `SYSTEM_PROMPT_QUIZ_FOLLOWUP`／`USER_PROMPT_COURSE_FOLLOWUP`：作答不佳則針對弱點追問，作答良好則改出新的不重複題目。
"""


@router.post(
    "/tab/quiz/llm-generate-followup",
    summary="Exam LLM Generate Follow-up Quiz",
    operation_id="exam_llm_generate_quiz_followup",
    description=_EXAM_LLM_GEN_FOLLOWUP_DESCRIPTION.strip(),
)
def exam_llm_generate_quiz_followup(
    request: Request,
    body: openapi_body(ExamLlmGenerateQuizFollowupRequest, _EXAM_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """依先前問答接續出下一題；寫入 follow_up 與 follow_up_exam_quiz_id。亦可改用 POST /exam/tab/quiz/create-llm-generate-followup 一次完成建立與接續出題。"""
    _ = request
    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=body.exam_quiz_id,
        rag_tab_id=body.rag_tab_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history_qa=body.quiz_history_list,
        follow_up_exam_quiz_id=body.follow_up_exam_quiz_id,
    )


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/create-llm-generate
# ---------------------------------------------------------------------------

_EXAM_CREATE_LLM_GENERATE_DESCRIPTION = """\
等同先 POST /exam/tab/quiz/create 再 POST /exam/tab/quiz/llm-generate。
Body 不需 `exam_quiz_id`（由 create 產生）；其餘 RAG 綁定、unit_type 出題邏輯與回應 JSON 同 `llm-generate`。
"""


@router.post(
    "/tab/quiz/create-llm-generate",
    summary="Exam Create Quiz and LLM Generate",
    operation_id="exam_create_llm_generate_quiz",
    description=_EXAM_CREATE_LLM_GENERATE_DESCRIPTION.strip(),
)
def exam_create_llm_generate_quiz(
    request: Request,
    body: openapi_body(ExamCreateLlmGenerateQuizRequest, _EXAM_CREATE_LLM_GENERATE_OPENAPI_EXAMPLES),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """先新增空白 Exam_Quiz，再 LLM 出題；回應同 llm-generate。"""
    _ = request
    try:
        created = _create_exam_quiz_record(
            exam_tab_id=body.exam_tab_id,
            caller_person_id=caller_person_id,
            course_id=course_id,
        )
        exam_quiz_id = int(created["exam_quiz_id"])
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"建立 Exam_Quiz 後無法取得 exam_quiz_id: {e!s}") from e
    except Exception as e:
        _logger.exception("POST /exam/tab/quiz/create-llm-generate 建立 Exam_Quiz 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=exam_quiz_id,
        rag_tab_id=body.rag_tab_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=False,
        quiz_history_stems=body.quiz_history_list,
    )


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/create-llm-generate-followup
# ---------------------------------------------------------------------------

_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_DESCRIPTION = """\
等同先 POST /exam/tab/quiz/create 再 POST /exam/tab/quiz/llm-generate-followup。
Body 不需 `exam_quiz_id`（由 create 產生）。
出題成功後**一律**寫入本列 `follow_up=true`；`follow_up_exam_quiz_id` 以請求傳入為準（可為 0）。
`quiz_history_list` 非空時使用追問 LLM prompt，否則出題邏輯同一般 llm-generate。
"""


@router.post(
    "/tab/quiz/create-llm-generate-followup",
    summary="Exam Create Quiz and LLM Generate Follow-up",
    operation_id="exam_create_llm_generate_quiz_followup",
    description=_EXAM_CREATE_LLM_GENERATE_FOLLOWUP_DESCRIPTION.strip(),
)
def exam_create_llm_generate_quiz_followup(
    request: Request,
    body: openapi_body(
        ExamCreateLlmGenerateQuizFollowupRequest,
        _EXAM_CREATE_LLM_GENERATE_FOLLOWUP_OPENAPI_EXAMPLES,
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """先新增空白 Exam_Quiz，再依先前問答接續 LLM 出題；回應同 llm-generate-followup。"""
    _ = request
    try:
        created = _create_exam_quiz_record(
            exam_tab_id=body.exam_tab_id,
            caller_person_id=caller_person_id,
            course_id=course_id,
        )
        exam_quiz_id = int(created["exam_quiz_id"])
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"建立 Exam_Quiz 後無法取得 exam_quiz_id: {e!s}") from e
    except Exception as e:
        _logger.exception("POST /exam/tab/quiz/create-llm-generate-followup 建立 Exam_Quiz 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return _exam_llm_generate_quiz_impl(
        exam_quiz_id=exam_quiz_id,
        rag_tab_id=body.rag_tab_id,
        rag_unit_id=body.rag_unit_id,
        rag_quiz_id=body.rag_quiz_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
        followup=True,
        quiz_history_qa=body.quiz_history_list,
        follow_up_exam_quiz_id=body.follow_up_exam_quiz_id,
        always_mark_follow_up=True,
    )


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/llm-grade
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/llm-grade", summary="Exam Grade Quiz", operation_id="exam_llm_grade_quiz")
@router.post("/tab/quiz/grade", summary="Exam Grade Quiz", include_in_schema=False)
async def exam_grade_submission(
    request: Request,
    background_tasks: BackgroundTasks,
    body: openapi_body(
        ExamQuizGradeRequest,
        {"exam_quiz_id": 1, "quiz_content": "", "quiz_answer": "學生作答文字"},
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    以 exam_quiz_id 定位題目，進行 RAG+LLM 非同步評分。
    unit_type 2／3／4 時改以 transcript 純 LLM 批改。
    評分 prompt 模板優先用 Exam_Quiz.quiz_user_prompt_text／answer_user_prompt_text（與 POST …/llm-generate 寫入一致），欄位為空時再讀 Rag_Quiz。
    評分完成後直接更新 Exam_Quiz.answer_content / answer_critique。
    回傳 202 + job_id；輪詢 GET /exam/tab/quiz/grade-result/{job_id}。
    """
    supabase = get_supabase()

    qsel = (
        supabase.table("Exam_Quiz")
        .select(
            "exam_quiz_id, exam_tab_id, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, course_id, unit_name, quiz_name, "
            "quiz_user_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, quiz_rate, "
            "answer_user_prompt_text, answer_content, answer_critique, updated_at, created_at"
        )
        .eq("exam_quiz_id", body.exam_quiz_id)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if not qsel.data or len(qsel.data) == 0:
        return JSONResponse(status_code=404, content={"error": f"找不到 exam_quiz_id={body.exam_quiz_id} 的 Exam_Quiz"})
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    rag_unit_id_val = qrow.get("rag_unit_id")
    stored_quiz_content = (qrow.get("quiz_content") or "").strip()

    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Exam_Quiz 評分"})

    quiz_content = (body.quiz_content or "").strip() or stored_quiz_content

    api_key = get_llm_api_key()
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "error": "請設定 LLM API Key：環境變數 LLM_API_KEY 或 OPENAI_API_KEY（本機可寫入 .env）",
            },
        )

    try:
        rag_uid_int = int(rag_unit_id_val) if rag_unit_id_val is not None else 0
    except (TypeError, ValueError):
        rag_uid_int = 0

    rag_rqid_int = 0
    _erq0 = qrow.get("rag_quiz_id")
    if _erq0 is not None:
        try:
            rag_rqid_int = int(_erq0)
            if rag_rqid_int < 0:
                rag_rqid_int = 0
        except (TypeError, ValueError):
            rag_rqid_int = 0

    grade_unit_filter: str | None = (qrow.get("unit_name") or "").strip() or None
    exam_grade_unit_type = 0
    transcript_for_unit = ""
    if rag_uid_int > 0:

        def _grade_unit_sel(cols: str, with_course_filter: bool):
            c = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(c)
                .eq("rag_unit_id", rag_uid_int)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        def _grade_unit_execute(cols: str):
            return execute_with_course_id_fallback(
                "Rag_Unit",
                lambda wc: _grade_unit_sel(cols, wc),
                course_id,
            )

        _gcols_full = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _gcols_no_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _gcols_no_fc = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _gcols_min = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _gcols_legacy_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        _gcols_no_fc_legacy_tr = (
            "rag_unit_id, rag_tab_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        try:
            unit_sel = _grade_unit_execute(_gcols_full)
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code != "42703":
                raise
            if "transcript" in msg:
                try:
                    unit_sel = _grade_unit_execute(_gcols_legacy_tr)
                except APIError as e_legacy:
                    if e_legacy.code == "42703" and "transcription" in (e_legacy.message or "").lower():
                        try:
                            unit_sel = _grade_unit_execute(_gcols_no_tr)
                        except APIError as e2:
                            if e2.code == "42703" and "folder_combination" in (e2.message or "").lower():
                                unit_sel = _grade_unit_execute(_gcols_min)
                            else:
                                raise
                    else:
                        raise
            elif "folder_combination" in msg:
                try:
                    unit_sel = _grade_unit_execute(_gcols_no_fc)
                except APIError as e2:
                    if e2.code == "42703" and "transcript" in (e2.message or "").lower():
                        try:
                            unit_sel = _grade_unit_execute(_gcols_no_fc_legacy_tr)
                        except APIError as e3:
                            if e3.code == "42703" and "transcription" in (e3.message or "").lower():
                                unit_sel = _grade_unit_execute(_gcols_min)
                            else:
                                raise
                    else:
                        raise
            else:
                raise
        if unit_sel.data:
            u0 = unit_sel.data[0]
            path_key = (u0.get("folder_combination") or u0.get("unit_name") or "").strip()
            if path_key:
                grade_unit_filter = path_key
            try:
                exam_grade_unit_type = int(u0.get("unit_type") or 0)
            except (TypeError, ValueError):
                exam_grade_unit_type = 0
            transcript_for_unit = transcript_from_row(u0)

    rag_id_used: int | None = None
    rt_exam = (str(qrow.get("rag_tab_id") or "").strip())
    if rt_exam:
        rag_id_used = rag_id_from_rag_tab_id(supabase, rt_exam, course_id)

    if rag_id_used is None or rag_id_used <= 0:
        rag_id_used, _ = resolve_exam_content_rag_id(
            supabase,
            request,
            stem_rag_unit_id=rag_uid_int if rag_uid_int > 0 else None,
            rag_quiz_id=rag_rqid_int if rag_rqid_int > 0 else None,
            course_id=course_id,
        )

    if rag_id_used is None or rag_id_used <= 0:
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    "無法決定評分用之 RAG：請確認 Exam_Quiz 填有對應 `Rag` 之 rag_tab_id，"
                    "或 rag_unit_id／rag_quiz_id 可自 Rag_Unit／Rag_Quiz 解析；仍可於 System_Setting "
                    "設定 rag_localhost／rag_deploy，value=Rag.rag_id。"
                ),
            },
        )

    try:
        rag_id = int(rag_id_used)
        row_exam, _stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
            supabase,
            rag_id,
            include_row=True,
            unit_name=grade_unit_filter,
            rag_unit_id=rag_uid_int if rag_uid_int > 0 else None,
        )
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    assert_row_course_id(row_exam, course_id, "Rag")

    transcript_text = (transcript_for_unit or instruction_from_rag_row(row_exam)).strip()

    quiz_user_prompt_exam = (qrow.get("quiz_user_prompt_text") or "").strip()
    answer_user_prompt_exam = (qrow.get("answer_user_prompt_text") or "").strip()
    exam_rag_quiz_id: int | None = rag_rqid_int if rag_rqid_int > 0 else None
    try:
        need_quiz = not quiz_user_prompt_exam
        need_answer = not answer_user_prompt_exam
        if (need_quiz or need_answer) and rag_rqid_int > 0:
            def build_rqgx(with_course_filter: bool):
                cols = select_without_course_id_if_needed(
                    "Rag_Quiz",
                    "rag_quiz_id, rag_tab_id, rag_unit_id, person_id, course_id, quiz_name, quiz_user_prompt_text, "
                    "quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
                    with_course_filter,
                )
                q = (
                    supabase.table("Rag_Quiz")
                    .select(cols)
                    .eq("rag_quiz_id", rag_rqid_int)
                    .eq("deleted", False)
                )
                if with_course_filter and course_id is not None:
                    q = q.eq("course_id", course_id)
                return q.limit(1)

            rqgx = execute_with_course_id_fallback("Rag_Quiz", build_rqgx, course_id)
            if rqgx.data:
                r0 = rqgx.data[0]
                if need_quiz:
                    quiz_user_prompt_exam = (r0.get("quiz_user_prompt_text") or "").strip()
                if need_answer:
                    answer_user_prompt_exam = (r0.get("answer_user_prompt_text") or "").strip()
    except (TypeError, ValueError):
        pass

    transcript_grade: str | None = None

    if exam_grade_unit_type in (2, 3, 4):
        if not transcript_text:
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcript 未設定（單元 2／3／4）；請於 Rag_Unit 或 Rag 設定 transcript"},
            )
        transcript_grade = transcript_text
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_exam_grade_tx_"))
    else:
        rag_zip_path = get_zip_path(rag_zip_tab_id)
        if not rag_zip_path or not rag_zip_path.exists():
            return JSONResponse(status_code=404, content={"error": f"找不到 RAG ZIP（tab_id={rag_zip_tab_id}）"})
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_exam_grade_"))
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
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    exam_quiz_id_int = int(body.exam_quiz_id)
    def insert_fn(rd, qa):
        return update_exam_quiz_with_grade(rd, qa, exam_quiz_id=exam_quiz_id_int)
    background_tasks.add_task(
        run_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        body.quiz_answer or "",
        _exam_grade_job_results,
        insert_fn,
        answer_user_prompt_exam,
        exam_quiz_id=exam_quiz_id_int,
        rag_quiz_id=exam_rag_quiz_id,
        unit_type=exam_grade_unit_type,
        transcript_grade=transcript_grade,
        quiz_user_prompt_text=quiz_user_prompt_exam,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


# ---------------------------------------------------------------------------
# GET /exam/tab/quiz/grade-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/tab/quiz/grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str, _person_id: PersonId, course_id: CourseId):
    """
    輪詢 Exam 評分結果（搭配 POST /exam/tab/quiz/llm-grade）。
    status: pending | ready | error；ready 時 result 含 quiz_comments、exam_quiz_id。
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
    out: dict[str, Any] = {"status": data["status"], "result": data.get("result"), "error": data.get("error")}
    if data["status"] == "ready":
        res = data.get("result")
        if isinstance(res, dict):
            eid = res.get("exam_quiz_id")
            if eid is not None:
                try:
                    eid_int = int(eid)
                    if eid_int > 0:
                        supabase = get_supabase()
                        q = (
                            supabase.table("Exam_Quiz")
                            .select("*")
                            .eq("exam_quiz_id", eid_int)
                            .eq("course_id", course_id)
                            .limit(1)
                            .execute()
                        )
                        if q.data:
                            out["exam_quiz"] = to_json_safe(q.data[0])
                except (TypeError, ValueError):
                    pass
    return out


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/rate
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/rate", summary="Exam Rate Quiz", status_code=200)
def update_exam_quiz_rate(
    body: openapi_body(ExamQuizRateRequest, {"exam_quiz_id": 1, "quiz_rate": 0}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """依 exam_quiz_id 更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。"""
    exam_quiz_id = int(body.exam_quiz_id)
    quiz_rate = int(body.quiz_rate)
    supabase = get_supabase()
    r = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, person_id, course_id")
        .eq("exam_quiz_id", exam_quiz_id)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={exam_quiz_id} 的 Exam_Quiz")
    qpid = (r.data[0].get("person_id") or "").strip()
    if qpid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權更新該題 quiz_rate")
    supabase.table("Exam_Quiz").update(
        {"quiz_rate": quiz_rate, "updated_at": now_taipei_iso()}
    ).eq("exam_quiz_id", exam_quiz_id).execute()
    after = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, quiz_rate, updated_at, created_at")
        .eq("exam_quiz_id", exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not after.data or len(after.data) == 0:
        raise HTTPException(status_code=500, detail="更新 quiz_rate 後讀取失敗")
    out = dict(after.data[0])
    out["message"] = "已更新 quiz_rate"
    return to_json_safe(out)
