"""
Exam API 模組。對應 public.Exam、public.Exam_Quiz。

檔案結構（由上而下）：模型／檢索說明 → LLM Prompt → 型別與作業快取 → Pydantic → 輔助函式與路由。

- GET /exam/tabs：列出 Exam（deleted=false，person_id 篩選，local 篩選），每筆帶 units（依 unit_name 分群之 Exam_Quiz）。
- GET /exam/rag-for-exams：列出 for_exam 測驗用 RAG 資料（Rag_Unit.for_exam=true 或含 Rag_Quiz.for_exam=true）。
- POST /exam/tab/create：建立一筆 Exam。
- PUT /exam/tab/tab-name：更新既有 Exam 的 tab_name。
- POST /exam/tab/quiz/create：新增空白 Exam_Quiz（不呼叫 LLM）。
- POST /exam/tab/quiz/llm-generate：出題請求須含 `rag_tab_id` 與單／題鍵；自該 RAG Tab 載入資料，不依賴 System_Setting。
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
from dependencies.person_id import PersonId
from fastapi.responses import JSONResponse, Response
from postgrest.exceptions import APIError
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from services.exam_queries import (
    all_exam_quizzes,
    ensure_exam_quiz_rag_id_keys,
    enrich_exam_quizzes_rag_tab_from_units,
    exam_default_row,
    exams_by_tab_ids,
    exams_table_select,
    group_exam_quizzes_into_units,
    quizzes_by_exam_tab_ids,
    quizzes_by_person_id,
    rag_quiz_for_exam_response_row,
    select_rag_row_with_transcription_fallback,
)
from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_exam_quiz_with_grade,
)
from utils.datetime_utils import now_taipei_iso, to_taipei_iso
from utils.json_utils import to_json_safe
from utils.llm_api_key_utils import get_llm_api_key
from utils.rag_exam_setting import is_localhost_request, rag_id_from_rag_tab_id, resolve_exam_content_rag_id
from utils.rag_stem_utils import get_rag_stem_from_rag_id, instruction_from_rag_row
from utils.supabase_client import get_supabase
from utils.zip_storage import generate_tab_id, get_zip_path

router = APIRouter(prefix="/exam", tags=["exam"])


# ---------------------------------------------------------------------------
# 模型與檢索常數
# ---------------------------------------------------------------------------
# 本模組不直接宣告 OpenAI 模型名；出題呼叫 `utils.quiz_generation`（`QUIZ_LLM_MODEL`、embedding、k），
# 批改呼叫 `services.grading`（`GRADE_LLM_MODEL`、檢索與 chunk 常數）。


# ---------------------------------------------------------------------------
# LLM Prompt 範本（exam 出題 user 前綴；置於課程內容／檢索片段之前）
# ---------------------------------------------------------------------------
# 由 _exam_llm_generate_api_instruction 以 .format 代入；與 grade 路由「僅 quiz_user_prompt」不同，
# exam 須把本次請求相關 id／名稱一併給模型，方便對齊寫回之 Exam_Quiz 列。
# rag_quiz_md：已含反引號或「（請求未傳入）」字面字串，故模板內不再加反引號包裹。

PROMPT_EXAM_LLM_GENERATE_USER_PREFIX = textwrap.dedent("""
    ## 本次請求 API 參數

    請一併納入出題考量（順序對齊 public.Exam_Quiz 請求／關聯欄）：

    - **exam_quiz_id**：`{exam_quiz_id}`
    - **unit_name**：{unit_name_md}
    - **rag_unit_id**：`{rag_unit_id}`
    - **rag_quiz_id**：{rag_quiz_md}
    - **quiz_name**：{quiz_name_md}

    ### quiz_user_prompt_text

    {quiz_user_prompt}
    """).strip()


ExamQuizRateValue = Literal[-1, 0, 1]

_exam_grade_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ListExamResponse(BaseModel):
    """GET /exam/tabs 回應：每筆 Exam 含 units（依 unit_name 分群），每個 unit 含 quizzes。"""
    exams: list[dict] = Field(
        ...,
        description="每筆 Exam 的 units[] 為 { unit_name, rag_unit_id?, quizzes[] }；quizzes[] 為 Exam_Quiz（含答案欄位）",
    )
    count: int


class ListRagForExamsResponse(BaseModel):
    """GET /exam/rag-for-exams：單元為完整 Rag_Unit；quizzes 含 RAG 關聯鍵與出題／批改 prompt。"""
    units: list[dict] = Field(
        ...,
        description="Rag_Unit 列；每筆 quizzes[] 含 rag_quiz_id、rag_tab_id、rag_unit_id、quiz_name、quiz_user_prompt_text、answer_user_prompt_text",
    )
    count: int


class CreateExamRequest(BaseModel):
    """POST /exam/tab/create：欄位同 public.Exam（exam_tab_id, tab_name, person_id, local）。"""
    exam_tab_id: str | None = Field(None, description="選填；未傳則由後端產生")
    tab_name: str = Field("", description="測驗顯示名稱")
    person_id: str = Field("", description="選填，寫入 Exam.person_id")
    local: bool = Field(False, description="是否為本機 Exam")


class UpdateExamUnitNameRequest(BaseModel):
    """PUT /exam/tab/tab-name：以 exam_id（主鍵）更新 tab_name。"""
    exam_id: int = Field(..., description="Exam 主鍵")
    tab_name: str = Field(..., description="新的顯示名稱")


class ExamCreateQuizRequest(BaseModel):
    """POST /exam/tab/quiz/create：新增空白 Exam_Quiz（無 LLM）。僅 exam_tab_id（不傳 rag_unit_id）。"""
    exam_tab_id: str = Field("", description="目標 Exam 的 exam_tab_id")


class ExamLlmGenerateQuizRequest(BaseModel):
    """POST /exam/tab/quiz/llm-generate：`exam_quiz_id`、`rag_tab_id`、`rag_unit_id`、`rag_quiz_id` 皆必填。
    請求中之 `rag_unit_id`、`rag_quiz_id` 所隸 `rag_tab_id` 須與 `rag_tab_id` 相符；Exam_Quiz 列鎖鍵規則同後述說明。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "exam_quiz_id": 1,
                "rag_tab_id": "string",
                "rag_unit_id": 1,
                "rag_quiz_id": 1,
            },
        },
    )

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
        description="Rag_Quiz 主鍵（>0），出題 user prompt 由此讀取。列鎖鍵規則同 rag_unit_id",
    )


class ExamQuizRateRequest(BaseModel):
    """POST /exam/tab/quiz/rate：更新 Exam_Quiz.quiz_rate。"""
    exam_quiz_id: int = Field(..., ge=1, description="Exam_Quiz 主鍵")
    quiz_rate: ExamQuizRateValue = Field(0, description="僅 -1、0、1")


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/tab/quiz/llm-grade：欄位順序對齊 public.Exam_Quiz（題幹優先於作答）；answer_user_prompt_text 僅自 Rag_Quiz 讀取。"""
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

def _load_exam_for_quiz(
    supabase: Any,
    *,
    exam_id: int,
    exam_tab_id: str,
    caller_person_id: str,
) -> tuple[str, str]:
    """回傳 (exam_tab_id, person_id)。需擇一傳入 exam_id 或 exam_tab_id。"""
    et = (exam_tab_id or "").strip()
    if exam_id > 0:
        exam_rows = (
            supabase.table("Exam")
            .select("exam_id, exam_tab_id, person_id")
            .eq("exam_id", exam_id)
            .eq("deleted", False)
            .execute()
        )
    elif et and et != "0":
        exam_rows = (
            supabase.table("Exam")
            .select("exam_id, exam_tab_id, person_id")
            .eq("exam_tab_id", et)
            .eq("deleted", False)
            .execute()
        )
    else:
        raise HTTPException(status_code=400, detail="請傳入 exam_id 或 exam_tab_id")
    if not exam_rows.data or len(exam_rows.data) == 0:
        raise HTTPException(status_code=404, detail="找不到對應的 Exam 資料")
    row = exam_rows.data[0]
    out_tab = (row.get("exam_tab_id") or "").strip()
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該 Exam 的 person_id 為空")
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權操作該 Exam")
    return out_tab, person_id


def _exam_llm_generate_api_instruction(
    *,
    exam_quiz_id: int,
    rag_unit_id: int,
    rag_quiz_id: int | None,
    unit_name: str | None,
    quiz_name: str | None,
    quiz_user_prompt_resolved: str,
) -> str:
    """
    組出 POST /exam/tab/quiz/llm-generate 送進 utils.generate_quiz* 的 quiz_user_prompt_text 前綴。

    quiz_user_prompt_resolved：自 Rag_Quiz（effective rag_quiz_id 來自 Exam_Quiz）解析之出題 prompt（可能空）。
    """
    rq = rag_quiz_id
    rq_md = f"`{rq}`" if rq is not None and rq > 0 else "（Exam_Quiz 未關聯 rag_quiz_id）"
    un = (unit_name or "").strip()
    qn = (quiz_name or "").strip()
    qup = (quiz_user_prompt_resolved or "").strip()
    return PROMPT_EXAM_LLM_GENERATE_USER_PREFIX.format(
        exam_quiz_id=exam_quiz_id,
        rag_unit_id=int(rag_unit_id or 0),
        rag_quiz_md=rq_md,
        unit_name_md=un if un else "（未提供）",
        quiz_name_md=qn if qn else "（未提供）",
        quiz_user_prompt=qup if qup else "（未提供）",
    )


# ---------------------------------------------------------------------------
# GET /exam/tabs
# ---------------------------------------------------------------------------

@router.get("/tabs", response_model=ListExamResponse)
def list_exams(
    request: Request,
    person_id: PersonId,
    local: bool | None = Query(
        None,
        description="僅回傳 Exam.local 與此值相同的列。未傳時：本機連線視為 true，否則 false",
    ),
):
    """列出 Exam（deleted=false，person_id 篩選，local 篩選）。每筆 Exam 帶 units（依 unit_name 分群的 Exam_Quiz）。"""
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = exams_table_select(exclude_deleted=True, local_match=local_filter)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        tab_ids = list(dict.fromkeys(
            str(r.get("exam_tab_id")) for r in data if r.get("exam_tab_id") is not None
        ))
        quizzes_by_tab = quizzes_by_exam_tab_ids(tab_ids)
        flat_qz = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_qz)
        ensure_exam_quiz_rag_id_keys(flat_qz)

        for row in data:
            tab_id = str(row.get("exam_tab_id") or "")
            row["units"] = group_exam_quizzes_into_units(quizzes_by_tab.get(tab_id, []))

        data = to_json_safe(data)
        return ListExamResponse(exams=data, count=len(data))
    except Exception as e:
        logging.exception("GET /exam/tabs 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Exam 失敗: {e!s}")


# ---------------------------------------------------------------------------
# GET /exam/rag-for-exams
# ---------------------------------------------------------------------------

@router.get(
    "/rag-for-exams",
    response_model=ListRagForExamsResponse,
    summary="List RAG units & quizzes marked for exam",
)
def list_rag_for_exams(_person_id: PersonId):
    """
    回傳 for-exam 相關 RAG 單元與題目（不限 person_id）：
    - 單元：Rag_Unit.deleted=false 且（Rag_Unit.for_exam=true 或至少一筆 Rag_Quiz.for_exam=true 隸屬該 rag_unit_id）。
    - quizzes：僅 Rag_Quiz.for_exam=true 且 deleted=false。
    """
    try:
        supabase = get_supabase()

        quizzes_resp = (
            supabase.table("Rag_Quiz")
            .select("rag_quiz_id, rag_tab_id, rag_unit_id, quiz_name, quiz_user_prompt_text, answer_user_prompt_text")
            .eq("for_exam", True)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
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
            units_flag_resp = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id")
                .eq("for_exam", True)
                .eq("deleted", False)
                .execute()
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
                logging.warning("Rag_Unit 無 for_exam 欄位；GET /exam/rag-for-exams 僅列出含 Rag_Quiz.for_exam 之單元")
            else:
                raise

        all_unit_ids = list(dict.fromkeys(list(unit_ids_from_units | unit_ids_from_quizzes)))
        if not all_unit_ids:
            return ListRagForExamsResponse(units=[], count=0)

        units = (
            supabase.table("Rag_Unit")
            .select("*")
            .in_("rag_unit_id", all_unit_ids)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
            .data or []
        )
        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        out = to_json_safe(units)
        return ListRagForExamsResponse(units=out, count=len(out))
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("GET /exam/rag-for-exams 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 for_exam RAG 失敗: {e!s}")


# ---------------------------------------------------------------------------
# POST /exam/tab/create
# ---------------------------------------------------------------------------

@router.post("/tab/create")
def create_exam(body: CreateExamRequest, caller_person_id: PersonId):
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
        .insert(exam_default_row(fid, tab_name=tab_name, person_id=person_id, local=body.local))
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
        "local": row.get("local", body.local),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


# ---------------------------------------------------------------------------
# PUT /exam/tab/tab-name
# ---------------------------------------------------------------------------

@router.put("/tab/tab-name", summary="Update Exam Tab Name")
def update_exam_unit_tab_name(body: UpdateExamUnitNameRequest, caller_person_id: PersonId):
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
            .select("exam_id, exam_tab_id, person_id")
            .eq("exam_id", body.exam_id)
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
        return {"exam_id": body.exam_id, "exam_tab_id": fid, "person_id": pid, "tab_name": tab_name, "updated_at": ts}
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
    exam_tab_id: str = PathParam(..., description="要刪除的 Exam 的 exam_tab_id"),
):
    """PUT /exam/tab/delete/{exam_tab_id}。軟刪除：將 Exam 的 deleted 設為 true。"""
    fid = (exam_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 exam_tab_id")
    supabase = get_supabase()
    r = supabase.table("Exam").select("exam_id, person_id").eq("exam_tab_id", fid).eq("deleted", False).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="找不到該 exam_tab_id 的 Exam 資料，或已刪除")
    pid = (r.data[0].get("person_id") or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權刪除該 Exam")
    supabase.table("Exam").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("exam_tab_id", fid).eq("deleted", False).execute()
    return {"message": "已將 Exam 標記為刪除", "exam_tab_id": fid, "person_id": pid}


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/create
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/create", summary="Exam Create Quiz (no LLM)", operation_id="exam_create_quiz")
def exam_insert_empty_quiz(body: ExamCreateQuizRequest, caller_person_id: PersonId):
    """新增一筆空白 Exam_Quiz，不呼叫 LLM；body 僅需 exam_tab_id。"""
    supabase = get_supabase()
    exam_tab_id, person_id = _load_exam_for_quiz(
        supabase,
        exam_id=0,
        exam_tab_id=body.exam_tab_id,
        caller_person_id=caller_person_id,
    )
    qts = now_taipei_iso()
    quiz_row: dict[str, Any] = {
        "exam_tab_id": exam_tab_id,
        "unit_name": "",
        "rag_unit_id": None,
        "rag_quiz_id": None,
        "person_id": person_id,
        "quiz_name": "",
        "quiz_content": "",
        "quiz_hint": "",
        "quiz_answer_reference": "",
        "answer_content": None,
        "answer_critique": None,
        "quiz_rate": 0,
        "created_at": qts,
        "updated_at": qts,
    }
    try:
        ins = supabase.table("Exam_Quiz").insert(quiz_row).execute()
        if not ins.data or len(ins.data) == 0:
            raise HTTPException(status_code=500, detail="寫入 Exam_Quiz 失敗（無回傳資料）")
        row = dict(ins.data[0])
        enrich_exam_quizzes_rag_tab_from_units([row])
        ensure_exam_quiz_rag_id_keys([row])
        return to_json_safe(row)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("POST /exam/tab/quiz/create 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/llm-generate
# ---------------------------------------------------------------------------

_EXAM_LLM_GEN_DESCRIPTION = """\
Body：`exam_quiz_id`、`rag_tab_id`、`rag_unit_id`、`rag_quiz_id` 皆必填。
`rag_tab_id` 須對應 `public.Rag.rag_tab_id`，且與所列 `rag_unit_id`、`rag_quiz_id` 在 DB 上所隸屬之 Tab 一致；並用此載入 ZIP／單元（**不依賴** System_Setting 之 `rag_localhost`/`rag_deploy`）。
若該 Exam_Quiz 列**已有**有效的 `rag_unit_id`、`rag_quiz_id`，請求兩鍵須與列**完全一致**，否則 400。
若列**尚未**寫入（缺其一或為 0），則以此請求綁定，出題成功後一併寫回。
`quiz_user_prompt_text` 僅自 Rag_Quiz（請求中的 `rag_quiz_id`）讀取，不另由 body 帶入文字。
unit_type 1（rag）時僅依 RAG ZIP／向量檢索出題，不注入 transcription。
unit_type 2／3／4 時不載入 RAG ZIP，改以 transcription 純 LLM 出題。
出題成功後更新該筆 Exam_Quiz（`rag_tab_id`、`quiz_name`、quiz_content／quiz_hint／quiz_answer_reference、rag_unit_id、rag_quiz_id；清空作答欄位）。
"""

@router.post(
    "/tab/quiz/llm-generate",
    summary="Rag LLM Generate Quiz",
    operation_id="exam_llm_generate_quiz",
    description=_EXAM_LLM_GEN_DESCRIPTION.strip(),
)
@router.post("/generate-quiz", include_in_schema=False)
def exam_llm_generate_quiz(request: Request, body: ExamLlmGenerateQuizRequest, caller_person_id: PersonId):
    """實作與說明見模組常數 `_EXAM_LLM_GEN_DESCRIPTION`（OpenAPI operation description）。"""
    supabase = get_supabase()
    qsel = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, exam_tab_id, unit_name, rag_tab_id, rag_unit_id, rag_quiz_id, person_id")
        .eq("exam_quiz_id", body.exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not qsel.data or len(qsel.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={body.exam_quiz_id} 的 Exam_Quiz")
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Exam_Quiz 出題")

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

    body_ruid = int(body.rag_unit_id)
    body_rqid = int(body.rag_quiz_id)

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

    def _rag_quiz_user_prompt_from_db(rag_quiz_id: int, *, explicit_in_request: bool) -> str:
        sel = (
            supabase.table("Rag_Quiz")
            .select("quiz_user_prompt_text")
            .eq("rag_quiz_id", rag_quiz_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if sel.data:
            return (sel.data[0].get("quiz_user_prompt_text") or "").strip()
        if explicit_in_request:
            raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={rag_quiz_id} 的 Rag_Quiz，或已刪除")
        return ""

    cand_rag_qid = effective_rqid

    tab_strip = (body.rag_tab_id or "").strip()
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

    ru_one = (
        supabase.table("Rag_Unit")
        .select("unit_name, rag_tab_id")
        .eq("rag_unit_id", effective_ruid)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
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

    rq_one = (
        supabase.table("Rag_Quiz")
        .select("rag_tab_id")
        .eq("rag_quiz_id", effective_rqid)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rq_one.data:
        raise HTTPException(status_code=404, detail=f"找不到 rag_quiz_id={effective_rqid} 之 Rag_Quiz")
    rq_tab = (rq_one.data[0].get("rag_tab_id") or "").strip()
    if rq_tab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_tab_id 須與 rag_quiz_id 所隸 Rag Tab 一致；"
                f"請求為 {tab_strip!r}，Rag_Quiz 為 {rq_tab!r}"
            ),
        )

    rag_id_resolved = rag_id_from_rag_tab_id(supabase, tab_strip)
    if rag_id_resolved is None or rag_id_resolved <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 rag_tab_id={tab_strip!r} 對應之 Rag（deleted=false）",
        )

    quiz_user_prompt_resolved = _rag_quiz_user_prompt_from_db(
        cand_rag_qid,
        explicit_in_request=True,
    )

    unit_filter: str | None = (ru_one.data[0].get("unit_name") or "").strip() or None
    stem_rag_unit_id: int | None = effective_ruid if effective_ruid > 0 else None
    if not unit_filter:
        unit_filter = (qrow.get("unit_name") or "").strip() or None

    api_key = get_llm_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 LLM API Key：環境變數 LLM_API_KEY 或 OPENAI_API_KEY（本機可寫入 .env）",
        )

    rag_rows = select_rag_row_with_transcription_fallback(supabase, rag_id_resolved)
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id_resolved} 的 Rag 資料，或已刪除")
    rag_row = rag_rows.data[0]
    rag_id = int(rag_row.get("rag_id") or 0)
    rag_tab_id_for_units = (rag_row.get("rag_tab_id") or "").strip()

    stem, rag_zip_tab_id = get_rag_stem_from_rag_id(
        supabase, rag_id, unit_name=unit_filter, rag_unit_id=stem_rag_unit_id
    )

    selected: dict | None = None
    if rag_tab_id_for_units:
        try:
            unit_q = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id, unit_name, transcription, unit_type")
                .eq("rag_tab_id", rag_tab_id_for_units)
                .eq("deleted", False)
                .order("created_at", desc=False)
                .execute()
            )
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and "transcription" in msg:
                unit_q = (
                    supabase.table("Rag_Unit")
                    .select("rag_unit_id, unit_name, unit_type")
                    .eq("rag_tab_id", rag_tab_id_for_units)
                    .eq("deleted", False)
                    .order("created_at", desc=False)
                    .execute()
                )
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
                if (u.get("unit_name") or "").strip() == unit_filter:
                    selected = u
                    break

    transcription_text = ""
    if selected:
        transcription_text = (selected.get("transcription") or "").strip()
    if not transcription_text:
        transcription_text = instruction_from_rag_row(rag_row)

    try:
        unit_type_val = int(selected.get("unit_type") or 0) if selected else 0
    except (TypeError, ValueError):
        unit_type_val = 0

    if unit_type_val in (2, 3, 4) and not transcription_text:
        raise HTTPException(
            status_code=400,
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 設定 transcription，或經 POST /rag/tab/build-rag-zip 寫入",
        )

    prompt_rag_unit_id = int(selected.get("rag_unit_id") or 0) if selected else int(qrow.get("rag_unit_id") or 0)
    prompt_rag_qid = cand_rag_qid
    un_for_prompt = (qrow.get("unit_name") or "").strip() or None
    qn_for_prompt = (qrow.get("quiz_name") or "").strip() or None
    api_instr = _exam_llm_generate_api_instruction(
        exam_quiz_id=body.exam_quiz_id,
        rag_unit_id=prompt_rag_unit_id,
        rag_quiz_id=prompt_rag_qid,
        unit_name=un_for_prompt,
        quiz_name=qn_for_prompt,
        quiz_user_prompt_resolved=quiz_user_prompt_resolved,
    )

    path: Path | None = None
    try:
        from utils.quiz_generation import generate_quiz, generate_quiz_transcription_only

        if unit_type_val in (2, 3, 4):
            result = generate_quiz_transcription_only(
                api_key=api_key,
                transcription=transcription_text,
                quiz_user_prompt_text=api_instr,
            )
        else:
            path = get_zip_path(rag_zip_tab_id)
            if not path or not path.exists():
                raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（tab_id={rag_zip_tab_id}）")
            result = generate_quiz(
                path,
                api_key=api_key,
                quiz_user_prompt_text=api_instr,
            )
        result["transcription"] = "" if unit_type_val == 1 else transcription_text
        result["rag_output"] = {"rag_tab_id": stem, "unit_name": stem, "filename": f"{stem}.zip"}

        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_answer_reference") or "").strip()
        result["quiz_content"] = qc
        result["quiz_hint"] = qh
        result["quiz_answer_reference"] = qref
        result["exam_quiz_id"] = body.exam_quiz_id
        qts = now_taipei_iso()
        unit_name_for_display = (unit_filter or stem or "").strip()
        quiz_name = ((stem or "").strip() or unit_name_for_display or (qrow.get("quiz_name") or "").strip() or "")
        result["quiz_name"] = quiz_name
        result["quiz_user_prompt_text"] = quiz_user_prompt_resolved
        result["unit_name"] = unit_name_for_display
        quiz_update: dict[str, Any] = {
            "quiz_name": quiz_name,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "rag_tab_id": tab_strip,
            "rag_unit_id": int(body.rag_unit_id),
            "rag_quiz_id": int(body.rag_quiz_id),
            "answer_content": None,
            "answer_critique": None,
            "updated_at": qts,
        }
        result["rag_tab_id"] = tab_strip
        result["rag_unit_id"] = int(body.rag_unit_id)
        result["rag_quiz_id"] = int(body.rag_quiz_id)
        try:
            supabase.table("Exam_Quiz").update(quiz_update).eq("exam_quiz_id", body.exam_quiz_id).execute()
        except Exception as e:
            logging.exception("POST /exam/tab/quiz/llm-generate 寫入 Exam_Quiz 失敗")
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
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/llm-grade
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/llm-grade", summary="Exam Grade Quiz", operation_id="exam_llm_grade_quiz")
@router.post("/tab/quiz/grade", summary="Exam Grade Quiz", include_in_schema=False)
async def exam_grade_submission(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ExamQuizGradeRequest,
    caller_person_id: PersonId,
):
    """
    以 exam_quiz_id 定位題目，進行 RAG+LLM 非同步評分。
    unit_type 2／3／4 時改以 transcription 純 LLM 批改。
    評分完成後直接更新 Exam_Quiz.answer_content / answer_critique。
    回傳 202 + job_id；輪詢 GET /exam/tab/quiz/grade-result/{job_id}。
    """
    supabase = get_supabase()

    qsel = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, exam_tab_id, unit_name, rag_tab_id, rag_unit_id, rag_quiz_id, person_id, quiz_content")
        .eq("exam_quiz_id", body.exam_quiz_id)
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
    transcription_for_unit = ""
    if rag_uid_int > 0:
        try:
            unit_sel = (
                supabase.table("Rag_Unit")
                .select("unit_name, unit_type, transcription")
                .eq("rag_unit_id", rag_uid_int)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
        except APIError as e:
            msg = (e.message or "").lower()
            if e.code == "42703" and "transcription" in msg:
                unit_sel = (
                    supabase.table("Rag_Unit")
                    .select("unit_name, unit_type")
                    .eq("rag_unit_id", rag_uid_int)
                    .eq("deleted", False)
                    .limit(1)
                    .execute()
                )
            else:
                raise
        if unit_sel.data:
            u0 = unit_sel.data[0]
            db_un = (u0.get("unit_name") or "").strip()
            if db_un:
                grade_unit_filter = db_un
            try:
                exam_grade_unit_type = int(u0.get("unit_type") or 0)
            except (TypeError, ValueError):
                exam_grade_unit_type = 0
            transcription_for_unit = (u0.get("transcription") or "").strip()

    rag_id_used: int | None = None
    rt_exam = (str(qrow.get("rag_tab_id") or "").strip())
    if rt_exam:
        rag_id_used = rag_id_from_rag_tab_id(supabase, rt_exam)

    if rag_id_used is None or rag_id_used <= 0:
        rag_id_used, _ = resolve_exam_content_rag_id(
            supabase,
            request,
            stem_rag_unit_id=rag_uid_int if rag_uid_int > 0 else None,
            rag_quiz_id=rag_rqid_int if rag_rqid_int > 0 else None,
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

    transcription_text = (transcription_for_unit or instruction_from_rag_row(row_exam)).strip()

    quiz_user_prompt_exam = ""
    answer_user_prompt_exam = ""
    exam_rag_quiz_id: int | None = None
    try:
        if rag_rqid_int > 0:
            exam_rag_quiz_id = rag_rqid_int
            rqgx = (
                supabase.table("Rag_Quiz")
                .select("quiz_user_prompt_text, answer_user_prompt_text")
                .eq("rag_quiz_id", rag_rqid_int)
                .eq("deleted", False)
                .limit(1)
                .execute()
            )
            if rqgx.data:
                r0 = rqgx.data[0]
                quiz_user_prompt_exam = (r0.get("quiz_user_prompt_text") or "").strip()
                answer_user_prompt_exam = (r0.get("answer_user_prompt_text") or "").strip()
    except (TypeError, ValueError):
        exam_rag_quiz_id = None

    transcription_grade: str | None = None

    if exam_grade_unit_type in (2, 3, 4):
        if not transcription_text:
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcription 未設定（單元 2／3／4）；請於 Rag_Unit 或 Rag 設定 transcription"},
            )
        transcription_grade = transcription_text
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
            try:
                rag_zip_path.unlink(missing_ok=True)
            except Exception:
                pass

    job_id = str(uuid.uuid4())
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    exam_quiz_id_int = int(body.exam_quiz_id)
    insert_fn = lambda rd, qa: update_exam_quiz_with_grade(rd, qa, exam_quiz_id=exam_quiz_id_int)
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
        transcription_grade=transcription_grade,
        quiz_user_prompt_text=quiz_user_prompt_exam,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


# ---------------------------------------------------------------------------
# GET /exam/tab/quiz/grade-result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/tab/quiz/grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str, _person_id: PersonId):
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
    return {"status": data["status"], "result": data.get("result"), "error": data.get("error")}


# ---------------------------------------------------------------------------
# POST /exam/tab/quiz/rate
# ---------------------------------------------------------------------------

@router.post("/tab/quiz/rate", summary="Exam Rate Quiz", status_code=200)
def update_exam_quiz_rate(body: ExamQuizRateRequest, caller_person_id: PersonId):
    """依 exam_quiz_id 更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。"""
    exam_quiz_id = int(body.exam_quiz_id)
    quiz_rate = int(body.quiz_rate)
    supabase = get_supabase()
    r = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, person_id")
        .eq("exam_quiz_id", exam_quiz_id)
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
        .select("exam_quiz_id, quiz_rate, updated_at")
        .eq("exam_quiz_id", exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not after.data or len(after.data) == 0:
        raise HTTPException(status_code=500, detail="更新 quiz_rate 後讀取失敗")
    out = dict(after.data[0])
    out["message"] = "已更新 quiz_rate"
    return to_json_safe(out)
