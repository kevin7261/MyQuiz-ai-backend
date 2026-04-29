"""
Exam API 模組。對應 public.Exam、public.Exam_Quiz。單元與 **RAG** 一致：`Exam_Quiz` 以 **`exam_tab_id`**、**`rag_unit_id`**（**Rag_Unit**）與 **`unit_name`** 描述所屬教材單元（**insert／update 僅使用目前 PostgREST 可見欄位**，不包含已自表內移除的 legacy 欄位）。
- GET /exam/tabs：列出 Exam（deleted=false，person_id 篩選，local 篩選），每筆帶 units（依 unit_name 分群之 Exam_Quiz；每筆 quiz 含 quiz_name、rag_quiz_id／rag_tab_id／rag_unit_id，答案欄位 answer_content／answer_critique，分數見 answer_critique 之 quiz_grade）。
- GET /exam/rag-for-exams：列出 for_exam 測驗用 RAG 資料。**包含** (1) `Rag_Unit.for_exam=true` 之單元（可無題）(2) 擁有任何 `Rag_Quiz.for_exam=true` 之 `rag_unit_id` 對應的單元。每單元之 `quizzes` 含 `rag_quiz_id`、`rag_tab_id`、`rag_unit_id`、`quiz_name`、`quiz_user_prompt_text`（出題）、`answer_user_prompt_text`（批改）；不限 person_id。
- POST /exam/tab/create：建立一筆 Exam（可傳 local，用法同 POST /rag/tab/create）。
- PUT /exam/tab/tab-name：更新既有 Exam 的 tab_name。
- POST /exam/tab/quiz/create：新增一筆空白 Exam_Quiz（**不呼叫 LLM**）；body 僅 **`exam_tab_id`**（**不需**傳 `rag_unit_id`）。單元與出題請於 **`POST /exam/tab/quiz/llm-generate`** 帶入 **`rag_unit_id`（>0）**／`unit_name` 等（對齊 RAG：`rag_unit_id`>0 時由 **Rag_Unit** 解析並寫入 `unit_name`、`rag_unit_id`）。
- POST /exam/tab/quiz/llm-generate：body **`exam_quiz_id`**、選填 **`rag_unit_id`（>0）**、**`rag_quiz_id`**、**`unit_name`**、**`quiz_name`**、**`quiz_user_prompt_text`**；上述 API 欄位會一併置入送 LLM 之 user 訊息（課程內容檢索之前）。`rag_unit_id`／`unit_name` 亦用於單元篩選並可寫回 Exam_Quiz；寫入 DB 者尚含 **quiz_name**、**quiz_content／quiz_hint／quiz_answer_reference**（`quiz_user_prompt_text` 不存 Exam_Quiz）。
- POST /exam/tab/quiz/llm-grade：body 含 **exam_quiz_id**、**quiz_answer**、選填 **quiz_content**、**answer_user_prompt_text**；此四者皆置入評分 prompt（未傳之文字欄位於 prompt 中標示為未提供）；非同步 RAG+LLM 評分；輪詢 GET /exam/tab/quiz/grade-result/{job_id}。
- POST /exam/tab/delete/{exam_tab_id}：軟刪除 Exam（deleted=true）。
- POST /exam/tab/quiz/rate：依 exam_quiz_id 更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。
"""

import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path as PathParam, Query, Request
from dependencies.person_id import PersonId
from fastapi.responses import JSONResponse, Response
from pydantic import AliasChoices, BaseModel, Field

from utils.datetime_utils import now_taipei_iso, to_taipei_iso
from utils.json_utils import to_json_safe
from utils.llm_api_key_utils import get_llm_api_key
from utils.rag_stem_utils import get_rag_stem_from_rag_id, instruction_from_rag_row
from utils.rag_exam_setting import fetch_exam_rag_id_from_settings, is_localhost_request
from utils.zip_storage import generate_tab_id, get_zip_path
from utils.supabase_client import get_supabase

from routers.grade import _run_grade_job_background, _update_exam_quiz_with_grade, _cleanup_grade_workspace

router = APIRouter(prefix="/exam", tags=["exam"])

ExamQuizRateValue = Literal[-1, 0, 1]


# --- 資料列建立輔助 ---

def _exam_default_row(
    exam_tab_id: str,
    *,
    tab_name: str = "",
    person_id: str = "",
    local: bool = False,
) -> dict[str, Any]:
    """Exam 表新增一筆時的預設欄位（不含 exam_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
    return {
        "exam_tab_id": exam_tab_id,
        "tab_name": tab_name,
        "person_id": person_id,
        "local": local,
        "deleted": False,
        "created_at": ts,
        "updated_at": ts,
    }


# --- 查詢輔助 ---

def _exams_table_select(
    exclude_deleted: bool = True,
    *,
    local_match: bool | None = None,
) -> list[dict]:
    """查詢 Exam 表。exclude_deleted=True 時僅回傳 deleted=False；依 created_at 升序。"""
    supabase = get_supabase()
    q = supabase.table("Exam").select("*")
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    q = q.order("created_at", desc=False)
    resp = q.execute()
    return resp.data or []


def _exams_by_ids(exam_ids: list[int]) -> list[dict]:
    """依 exam_id 查詢 Exam 表（僅 deleted=False）。"""
    if not exam_ids:
        return []
    supabase = get_supabase()
    resp = supabase.table("Exam").select("*").in_("exam_id", exam_ids).eq("deleted", False).execute()
    return resp.data or []


def _exams_by_tab_ids(exam_tab_ids: list[str]) -> list[dict]:
    """依 exam_tab_id 查詢 Exam 表（僅 deleted=False）。"""
    if not exam_tab_ids:
        return []
    supabase = get_supabase()
    resp = supabase.table("Exam").select("*").in_("exam_tab_id", exam_tab_ids).eq("deleted", False).execute()
    return resp.data or []


def _quizzes_by_exam_tab_ids(exam_tab_ids: list[str]) -> dict[str, list[dict]]:
    """依 exam_tab_id 查詢 Exam_Quiz，回傳 exam_tab_id -> list of quiz（依 created_at 升序）。"""
    if not exam_tab_ids:
        return {}
    supabase = get_supabase()
    resp = (
        supabase.table("Exam_Quiz")
        .select("*")
        .in_("exam_tab_id", exam_tab_ids)
        .order("created_at", desc=False)
        .execute()
    )
    rows = resp.data or []
    out: dict[str, list[dict]] = {tid: [] for tid in exam_tab_ids}
    for row in rows:
        tid = row.get("exam_tab_id")
        if tid is not None:
            out.setdefault(str(tid), []).append(row)
    return out


def _group_exam_quizzes_into_units(quizzes: list[dict]) -> list[dict]:
    """依 Exam_Quiz.unit_name 分群；units[] 每筆含 unit_name、rag_unit_id（取自該群首筆有效 rag_unit_id）、quizzes[]。"""
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for q in quizzes:
        un = (q.get("unit_name") or "").strip()
        if un not in buckets:
            order.append(un)
            buckets[un] = []
        buckets[un].append(q)
    units_out: list[dict] = []
    for un in order:
        qlist = buckets[un]
        rag_uid: int | None = None
        for q in qlist:
            ru = q.get("rag_unit_id")
            if ru is None:
                continue
            try:
                if int(ru) > 0:
                    rag_uid = int(ru)
                    break
            except (TypeError, ValueError):
                pass
        units_out.append({"unit_name": un, "rag_unit_id": rag_uid, "quizzes": qlist})
    return units_out


def _quizzes_by_person_id(person_id: str) -> list[dict]:
    """依 person_id 查詢 Exam_Quiz 全部筆數（供分析使用）。"""
    pid = (person_id or "").strip()
    if not pid:
        return []
    supabase = get_supabase()
    resp = supabase.table("Exam_Quiz").select("*").eq("person_id", pid).execute()
    return resp.data or []


def _all_exam_quizzes() -> list[dict]:
    """查詢 Exam_Quiz 表全部筆數（供 course analysis 使用）。"""
    supabase = get_supabase()
    resp = supabase.table("Exam_Quiz").select("*").execute()
    return resp.data or []


def _exam_quiz_rag_tab_id_str(q: dict) -> str:
    rt = q.get("rag_tab_id")
    if rt is None:
        return ""
    return str(rt).strip()


def _enrich_exam_quizzes_rag_tab_from_units(quizzes_flat: list[dict]) -> None:
    """Exam_Quiz 列若無 rag_tab_id 但有 rag_unit_id，自 Rag_Unit 補上 rag_tab_id（就地修改）。"""
    if not quizzes_flat:
        return
    missing_units: set[int] = set()
    for q in quizzes_flat:
        if _exam_quiz_rag_tab_id_str(q):
            continue
        ru = q.get("rag_unit_id")
        if ru is None:
            continue
        try:
            missing_units.add(int(ru))
        except (TypeError, ValueError):
            pass
    if not missing_units:
        return
    supabase = get_supabase()
    resp = (
        supabase.table("Rag_Unit")
        .select("rag_unit_id, rag_tab_id")
        .in_("rag_unit_id", list(missing_units))
        .eq("deleted", False)
        .execute()
    )
    tab_for_unit: dict[int, str | None] = {}
    for row in resp.data or []:
        rid = row.get("rag_unit_id")
        if rid is None:
            continue
        try:
            uid = int(rid)
        except (TypeError, ValueError):
            continue
        rtv = row.get("rag_tab_id")
        rt = (str(rtv).strip() if rtv is not None else "") or None
        tab_for_unit[uid] = rt
    for q in quizzes_flat:
        if _exam_quiz_rag_tab_id_str(q):
            continue
        ru = q.get("rag_unit_id")
        if ru is None:
            continue
        try:
            uid = int(ru)
        except (TypeError, ValueError):
            continue
        rt = tab_for_unit.get(uid)
        if rt:
            q["rag_tab_id"] = rt


def _ensure_exam_quiz_rag_id_keys(quizzes_flat: list[dict]) -> None:
    """確保每筆 Exam_Quiz dict 皆含 rag_quiz_id、rag_tab_id、rag_unit_id 鍵（無則 null）。"""
    for q in quizzes_flat:
        for key in ("rag_quiz_id", "rag_tab_id", "rag_unit_id"):
            if key not in q:
                q[key] = None


def exam_quiz_grade_from_critique(quiz: dict[str, Any]) -> int | None:
    """自 Exam_Quiz.answer_critique 解析 quiz_grade；無法解析則 None。"""
    raw = quiz.get("answer_critique")
    if raw is None:
        return None
    try:
        data: Any
        if isinstance(raw, dict):
            data = raw
        else:
            s = str(raw).strip()
            if not s:
                return None
            data = json.loads(s)
        if not isinstance(data, dict):
            return None
        g = data.get("quiz_grade")
        if g is None:
            meta = data.get("quiz_grade_metadata")
            if isinstance(meta, dict):
                g = meta.get("quiz_grade", meta.get("score"))
        if g is None:
            return None
        return int(round(float(g)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _rag_quiz_for_exam_response_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rag_Quiz 列：for-exam 端點回傳 RAG 關聯鍵與出題／批改 prompt。"""
    ru = row.get("rag_unit_id")
    rag_unit_id: int | None
    try:
        rag_unit_id = int(ru) if ru is not None else None
    except (TypeError, ValueError):
        rag_unit_id = None
    rq = row.get("rag_quiz_id")
    try:
        rag_quiz_id = int(rq) if rq is not None else None
    except (TypeError, ValueError):
        rag_quiz_id = None
    rt = row.get("rag_tab_id")
    rag_tab_id = (str(rt).strip() if rt is not None else "") or None
    return {
        "rag_quiz_id": rag_quiz_id,
        "rag_tab_id": rag_tab_id,
        "rag_unit_id": rag_unit_id,
        "quiz_name": str(row.get("quiz_name") or ""),
        "quiz_user_prompt_text": str(row.get("quiz_user_prompt_text") or ""),
        "answer_user_prompt_text": str(row.get("answer_user_prompt_text") or ""),
    }


# --- Pydantic models ---

class ListExamResponse(BaseModel):
    """GET /exam/tabs 回應：每筆 Exam 含 units（依 unit_name 分群），每個 unit 含 quizzes（Exam_Quiz，答案欄位內嵌）。"""
    exams: list[dict] = Field(
        ...,
        description="每筆 Exam 的 units[] 為 { unit_name, rag_unit_id?, quizzes[] }；quizzes[] 為 Exam_Quiz（含 quiz_name、rag_quiz_id／rag_tab_id／rag_unit_id、answer_content／answer_critique 等）",
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
    """POST /exam/tab/quiz/llm-generate；exam_quiz_id 必填；其餘選填。"""

    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（須已存在，通常為 create 後之錨點列）")
    rag_unit_id: int = Field(
        0,
        ge=0,
        description="選填；>0 時依 Rag_Unit 解析並寫入 Exam_Quiz.rag_unit_id／unit_name，並優先用於 RAG 單元篩選；0 表示此請求不更新 rag_unit_id",
    )
    rag_quiz_id: int | None = Field(
        None,
        description="選填；來源 Rag_Quiz 主鍵，寫入 Exam_Quiz.rag_quiz_id；請求未帶此欄時不改 DB；可傳 null 清空",
    )
    unit_name: str = Field(
        "",
        description="選填；非空時作為單元篩選並寫入 Exam_Quiz.unit_name（若未同時傳 rag_unit_id>0）；與 rag_unit_id>0 並傳時以 Rag_Unit 解析結果為準",
    )
    quiz_name: str = Field(
        "",
        description="顯示用測驗名稱；空字串則由 repack stem／單元 unit_name 推算；寫入 Exam_Quiz.quiz_name；並與其餘 body 欄位一併置入送 LLM 之 user 訊息",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="使用者出題補充（可空）；與其餘 body 欄位一併置入送 LLM 之 user 訊息；並置於 JSON 回應（未必寫入 Exam_Quiz）",
    )


class ExamQuizRateRequest(BaseModel):
    """POST /exam/tab/quiz/rate：更新 Exam_Quiz.quiz_rate。"""
    exam_quiz_id: int = Field(..., ge=1, description="Exam_Quiz 主鍵")
    quiz_rate: ExamQuizRateValue = Field(0, description="僅 -1、0、1")


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/tab/quiz/llm-grade：以 exam_quiz_id 定位題目；body 各欄皆置入評分 prompt。"""
    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（必填，>0）；置入評分 prompt")
    quiz_answer: str = Field(
        ...,
        description="學生作答；寫入 Exam_Quiz.answer_content；置入評分 prompt",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )
    quiz_content: str = Field(
        "",
        description="選填；若空則使用 Exam_Quiz 中存的 quiz_content；置入評分 prompt",
    )
    answer_user_prompt_text: str = Field(
        "",
        description="作答補充／批改指引（可空）；置入評分 prompt（未傳或空字串時 prompt 仍標示該欄）",
    )


# 非同步評分結果暫存
_exam_grade_job_results: dict[str, dict[str, Any]] = {}


# --- GET /exam/tabs ---

@router.get("/tabs", response_model=ListExamResponse)
def list_exams(
    request: Request,
    person_id: PersonId,
    local: bool | None = Query(
        None,
        description="僅回傳 Exam.local 與此值相同的列。未傳時：本機連線視為 true，否則 false",
    ),
):
    """
    列出 Exam（deleted=false，person_id 篩選，local 篩選）。
    每筆 Exam 帶 units（依 unit_name 分群），每個 unit 帶 quizzes（Exam_Quiz，含 quiz_name、rag_quiz_id／rag_tab_id／rag_unit_id 與 answer_content／answer_critique）。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _exams_table_select(exclude_deleted=True, local_match=local_filter)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        tab_ids = list(dict.fromkeys(
            str(r.get("exam_tab_id")) for r in data if r.get("exam_tab_id") is not None
        ))
        quizzes_by_tab = _quizzes_by_exam_tab_ids(tab_ids)
        flat_qz = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        _enrich_exam_quizzes_rag_tab_from_units(flat_qz)
        _ensure_exam_quiz_rag_id_keys(flat_qz)

        for row in data:
            tab_id = str(row.get("exam_tab_id") or "")
            qz_for_tab = quizzes_by_tab.get(tab_id, [])
            row["units"] = _group_exam_quizzes_into_units(qz_for_tab)

        data = to_json_safe(data)
        return ListExamResponse(exams=data, count=len(data))
    except Exception as e:
        logging.exception("GET /exam/tabs 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Exam 失敗: {e!s}")


# --- GET /exam/rag-for-exams ---

@router.get(
    "/rag-for-exams",
    response_model=ListRagForExamsResponse,
    summary="List RAG units & quizzes marked for exam",
)
def list_rag_for_exams(_person_id: PersonId):
    """
    回傳 for-exam 相關 RAG 單元與題目（**不限 person_id**）：

    - **單元**：`Rag_Unit.deleted=false` 且（`Rag_Unit.for_exam=true` **或** 至少一筆
      `Rag_Quiz.for_exam=true` 隸屬該 `rag_unit_id`）。若僅題目標記 for_exam、單元未標，仍會出現。
    - **quizzes**：僅 `Rag_Quiz.for_exam=true` 且 `deleted=false`；每筆回傳
      `rag_quiz_id`、`rag_tab_id`、`rag_unit_id`、`quiz_name`、`quiz_user_prompt_text`（出題 prompt）、`answer_user_prompt_text`（批改 prompt）。

    query `person_id` 仍必填（全站慣例），此端點不用於篩選。
    """
    try:
        supabase = get_supabase()

        quizzes_resp = (
            supabase.table("Rag_Quiz")
            .select(
                "rag_quiz_id, rag_tab_id, rag_unit_id, quiz_name, quiz_user_prompt_text, answer_user_prompt_text"
            )
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
            quizzes_by_unit.setdefault(uid, []).append(_rag_quiz_for_exam_response_row(q))

        units_flag_resp = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id")
            .eq("for_exam", True)
            .eq("deleted", False)
            .execute()
        )
        unit_ids_from_units: set[int] = set()
        for u in units_flag_resp.data or []:
            rid = u.get("rag_unit_id")
            if rid is not None:
                try:
                    unit_ids_from_units.add(int(rid))
                except (TypeError, ValueError):
                    pass

        all_unit_ids = list(dict.fromkeys(list(unit_ids_from_units | unit_ids_from_quizzes)))
        if not all_unit_ids:
            return ListRagForExamsResponse(units=[], count=0)

        units_resp = (
            supabase.table("Rag_Unit")
            .select("*")
            .in_("rag_unit_id", all_unit_ids)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
        units = units_resp.data or []

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


# --- POST /exam/tab/create ---

@router.post("/tab/create")
def create_exam(body: CreateExamRequest, caller_person_id: PersonId):
    """
    建立一筆 Exam。exam_tab_id 可選（未傳由後端產生）；local 選填（預設 false）。
    回傳 exam_id、exam_tab_id、person_id、tab_name、local、created_at。
    """
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
        .insert(
            _exam_default_row(fid, tab_name=tab_name, person_id=person_id, local=body.local)
        )
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


# --- PUT /exam/tab/tab-name ---

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
        return {
            "exam_id": body.exam_id,
            "exam_tab_id": fid,
            "person_id": pid,
            "tab_name": tab_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- POST /exam/tab/delete/{exam_tab_id} ---

@router.post("/tab/delete/{exam_tab_id}", status_code=200)
def delete_exam(
    caller_person_id: PersonId,
    exam_tab_id: str = PathParam(..., description="要刪除的 Exam 的 exam_tab_id"),
):
    """軟刪除：將 Exam 的 deleted 設為 true。"""
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


# --- Exam quiz：解析 Exam（create 與 llm-generate 共用）---

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


def _resolve_unit_name_and_rag_for_new_quiz(
    supabase: Any,
    *,
    unit_name: str,
    rag_unit_id: int,
) -> tuple[str, int | None]:
    """
    回傳 (unit_name, rag_unit_id)。
    擇一：rag_unit_id>0，或非空 unit_name（後者 rag_unit_id 為 None）。
    Exam_Quiz 表不存 rag_tab_id；GET 回應可經 _enrich_exam_quizzes_rag_tab_from_units 補上。
    """
    un = (unit_name or "").strip()
    if rag_unit_id > 0:
        ru_sel = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id, unit_name")
            .eq("rag_unit_id", rag_unit_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not ru_sel.data:
            raise HTTPException(status_code=404, detail=f"找不到 rag_unit_id={rag_unit_id} 的 Rag_Unit，或已刪除")
        ru_row = ru_sel.data[0]
        un_db = (ru_row.get("unit_name") or "").strip()
        resolved_name = un_db or un
        if not resolved_name:
            raise HTTPException(status_code=400, detail="Rag_Unit 無 unit_name，請在 body 傳入 unit_name")
        return resolved_name, rag_unit_id
    if un:
        return un, None
    raise HTTPException(
        status_code=400,
        detail="請傳入 rag_unit_id（>0），或非空 unit_name",
    )


def _exam_llm_generate_api_instruction(body: ExamLlmGenerateQuizRequest) -> str:
    """將 llm-generate 請求 body 各欄標示後置於送 LLM 之 user 訊息開頭（課程內容檢索之前）。"""
    rq = body.rag_quiz_id
    rq_line = f"rag_quiz_id: {rq}" if rq is not None else "rag_quiz_id: （請求未傳入）"
    un = (body.unit_name or "").strip()
    qn = (body.quiz_name or "").strip()
    qup = (body.quiz_user_prompt_text or "").strip()
    lines = [
        "【本次請求 API 參數（請一併納入出題考量）】",
        f"exam_quiz_id: {body.exam_quiz_id}",
        f"rag_unit_id: {int(body.rag_unit_id or 0)}",
        rq_line,
        f"unit_name: {un if un else '（未提供）'}",
        f"quiz_name: {qn if qn else '（未提供）'}",
        "quiz_user_prompt_text:",
        qup if qup else "（未提供）",
    ]
    return "\n".join(lines)


# --- POST /exam/tab/quiz/create（無 LLM；錨點列；單元於 llm-generate 對齊 Rag）---

@router.post("/tab/quiz/create", summary="Exam Create Quiz (no LLM)", operation_id="exam_create_quiz")
def exam_insert_empty_quiz(body: ExamCreateQuizRequest, caller_person_id: PersonId):
    """
    新增一筆空白 Exam_Quiz，**不呼叫 LLM**；body 僅需 **`exam_tab_id`**（**不需**上傳 `rag_unit_id`）。
    若要以 **`rag_unit_id`（>0）** 解析 **Rag_Unit** 並寫入 **`unit_name`、`rag_unit_id`**（對齊 `POST /rag/tab/unit/quiz/create` 之語意），請於 **`POST /exam/tab/quiz/llm-generate`** 帶入。
    """
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
        _enrich_exam_quizzes_rag_tab_from_units([row])
        _ensure_exam_quiz_rag_id_keys([row])
        return to_json_safe(row)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("POST /exam/tab/quiz/create 錯誤")
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- POST /exam/tab/quiz/llm-generate（對齊 POST /rag/tab/unit/quiz/llm-generate）---

@router.post("/tab/quiz/llm-generate", summary="Rag LLM Generate Quiz", operation_id="exam_llm_generate_quiz")
@router.post("/generate-quiz", include_in_schema=False)
def exam_llm_generate_quiz(request: Request, body: ExamLlmGenerateQuizRequest, caller_person_id: PersonId):
    """
    Body：**`exam_quiz_id`** 必填；選填 **`rag_unit_id`（>0）**、**`rag_quiz_id`**、**`unit_name`**、**`quiz_name`**、**`quiz_user_prompt_text`**。
    依系統測驗 RAG ZIP 由 LLM 出題後**更新**該筆 Exam_Quiz：**quiz_name**、**quiz_content／quiz_hint／quiz_answer_reference**、選填之 **unit_name／rag_unit_id／rag_quiz_id**（並清空作答欄位）。
    回傳 JSON 含 quiz_content、quiz_hint、quiz_reference_answer、exam_quiz_id、quiz_name、quiz_user_prompt_text、unit_name、rag_unit_id、rag_quiz_id 等。
    """
    supabase = get_supabase()
    qsel = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, exam_tab_id, unit_name, rag_unit_id, rag_quiz_id, person_id")
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

    body_un = (body.unit_name or "").strip()
    body_ruid = int(body.rag_unit_id or 0)

    effective_unit_name: str | None = None
    effective_rag_unit_id: int | None = None
    unit_filter: str | None = None

    if body_ruid > 0:
        resolved_u, resolved_ruid = _resolve_unit_name_and_rag_for_new_quiz(
            supabase,
            unit_name=body_un,
            rag_unit_id=body_ruid,
        )
        effective_unit_name = resolved_u
        effective_rag_unit_id = resolved_ruid
        unit_filter = (resolved_u or "").strip() or None
    elif body_un:
        effective_unit_name = body_un
        unit_filter = body_un
    else:
        unit_filter = (qrow.get("unit_name") or "").strip() or None
        if not unit_filter:
            rag_unit_val = qrow.get("rag_unit_id")
            if rag_unit_val is not None:
                try:
                    ruid = int(rag_unit_val)
                    if ruid > 0:
                        unit_sel = (
                            supabase.table("Rag_Unit")
                            .select("unit_name")
                            .eq("rag_unit_id", ruid)
                            .eq("deleted", False)
                            .limit(1)
                            .execute()
                        )
                        if unit_sel.data:
                            unit_filter = (unit_sel.data[0].get("unit_name") or "").strip() or None
                except (TypeError, ValueError):
                    pass

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
            detail="尚未設定供測驗用 RAG rag_id：請於 System_Setting 設定 key rag_localhost（本機）或 rag_deploy（非本機），value 為 Rag.rag_id",
        )
    rag_rows = (
        supabase.table("Rag")
        .select("rag_id, rag_tab_id, transcription")
        .eq("rag_id", rag_id_from_setting)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id_from_setting} 的 Rag 資料，或已刪除")
    rag_row = rag_rows.data[0]
    rag_id = int(rag_row.get("rag_id") or 0)
    rag_tab_id_for_units = (rag_row.get("rag_tab_id") or "").strip()

    stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id, unit_name=unit_filter)

    selected: dict | None = None
    if rag_tab_id_for_units:
        unit_q = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id, unit_name, transcription")
            .eq("rag_tab_id", rag_tab_id_for_units)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
        units = unit_q.data or []
        if not unit_filter:
            selected = units[0] if units else None
        else:
            for u in units:
                if (u.get("unit_name") or "").strip() == unit_filter:
                    selected = u
                    break
    transcription_text = ""
    if selected:
        transcription_text = (selected.get("transcription") or "").strip()
    if not transcription_text:
        transcription_text = instruction_from_rag_row(rag_row)
    if not transcription_text:
        raise HTTPException(
            status_code=400,
            detail="供測驗 RAG 的 transcription 未設定：請在該 Rag 或對應 Rag_Unit 設定 transcription；單元 2／3／4 可經 POST /rag/tab/build-rag-zip 寫入 Rag_Unit.transcription",
        )

    path = get_zip_path(rag_zip_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（tab_id={rag_zip_tab_id}）")

    try:
        from utils.quiz_generation import generate_quiz
        result = generate_quiz(
            path,
            api_key=api_key,
            transcription=transcription_text,
            user_instruction=_exam_llm_generate_api_instruction(body),
        )
        result["transcription"] = transcription_text
        result["rag_output"] = {
            "rag_tab_id": stem,
            "unit_name": stem,
            "filename": f"{stem}.zip",
        }

        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_reference_answer") or "").strip()
        result["quiz_content"] = qc
        result["quiz_hint"] = qh
        result["quiz_reference_answer"] = qref
        result["exam_quiz_id"] = body.exam_quiz_id
        qup = (body.quiz_user_prompt_text or "").strip()
        qts = now_taipei_iso()
        unit_name_for_display = (unit_filter or stem or "").strip()
        body_quiz_name = (body.quiz_name or "").strip()
        quiz_name = body_quiz_name or ((stem or "").strip() or unit_name_for_display or "")
        result["quiz_name"] = quiz_name
        result["quiz_user_prompt_text"] = qup
        result["unit_name"] = unit_name_for_display
        quiz_update: dict[str, Any] = {
            "quiz_name": quiz_name,
            "quiz_content": qc,
            "quiz_hint": qh,
            "quiz_answer_reference": qref,
            "answer_content": None,
            "answer_critique": None,
            "updated_at": qts,
        }
        if effective_unit_name is not None:
            quiz_update["unit_name"] = effective_unit_name
        if effective_rag_unit_id is not None:
            quiz_update["rag_unit_id"] = effective_rag_unit_id
        body_dump = body.model_dump(exclude_unset=True)
        if "rag_quiz_id" in body_dump:
            quiz_update["rag_quiz_id"] = body_dump["rag_quiz_id"]
        if "rag_unit_id" in quiz_update:
            result["rag_unit_id"] = quiz_update["rag_unit_id"]
        else:
            result["rag_unit_id"] = qrow.get("rag_unit_id")
        if "rag_quiz_id" in quiz_update:
            result["rag_quiz_id"] = quiz_update["rag_quiz_id"]
        else:
            result["rag_quiz_id"] = qrow.get("rag_quiz_id")
        try:
            supabase.table("Exam_Quiz").update(quiz_update).eq("exam_quiz_id", body.exam_quiz_id).execute()
        except Exception as e:
            logging.exception("POST /exam/tab/quiz/llm-generate 寫入 Exam_Quiz 失敗")
            raise HTTPException(status_code=500, detail=f"寫入 Exam_Quiz 失敗: {e!s}") from e
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# --- POST /exam/tab/quiz/llm-grade（舊路徑 /tab/quiz/grade 仍註冊、僅隱藏於 OpenAPI）---

@router.post("/tab/quiz/llm-grade", summary="Exam Grade Quiz", operation_id="exam_llm_grade_quiz")
@router.post("/tab/quiz/grade", summary="Exam Grade Quiz", include_in_schema=False)
async def exam_grade_submission(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ExamQuizGradeRequest,
    caller_person_id: PersonId,
):
    """
    以 exam_quiz_id 定位題目，進行 RAG+LLM 非同步評分（對齊 `POST /rag/tab/unit/quiz/llm-grade` 之 202 + job_id 流程）。
    Body 含 **`exam_quiz_id`**、**`quiz_answer`**、選填 **`quiz_content`**、**`answer_user_prompt_text`**（批改指引；併入評分 prompt）。
    評分完成後直接更新 Exam_Quiz.answer_content / answer_critique。
    回傳 202 與 job_id；輪詢 GET /exam/tab/quiz/grade-result/{job_id}。
    """
    supabase = get_supabase()

    # 查詢 Exam_Quiz
    qsel = (
        supabase.table("Exam_Quiz")
        .select("exam_quiz_id, exam_tab_id, unit_name, rag_unit_id, person_id, quiz_content")
        .eq("exam_quiz_id", body.exam_quiz_id)
        .limit(1)
        .execute()
    )
    if not qsel.data or len(qsel.data) == 0:
        return JSONResponse(status_code=404, content={"error": f"找不到 exam_quiz_id={body.exam_quiz_id} 的 Exam_Quiz"})
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    exam_tab_id = (qrow.get("exam_tab_id") or "").strip()
    rag_unit_id_val = qrow.get("rag_unit_id")
    stored_quiz_content = (qrow.get("quiz_content") or "").strip()

    if person_id != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權對該 Exam_Quiz 評分"})

    # 決定使用的 quiz_content
    quiz_content = (body.quiz_content or "").strip() or stored_quiz_content

    api_key = get_llm_api_key()
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "請先於系統設定（/system-settings/llm-api-key）填寫 LLM API Key"},
        )

    _, rag_id_from_setting = fetch_exam_rag_id_from_settings(supabase, request)
    if rag_id_from_setting is None or rag_id_from_setting <= 0:
        return JSONResponse(
            status_code=404,
            content={"error": "尚未設定供測驗用 RAG rag_id：請於 System_Setting 設定 key rag_localhost（本機）或 rag_deploy（非本機），value 為 Rag.rag_id"},
        )

    # 取得 unit_name（用於篩選 RAG ZIP）、unit_type（評分時與 ZIP 內容格式一致）
    grade_unit_filter: str | None = (qrow.get("unit_name") or "").strip() or None
    exam_grade_unit_type = 0
    try:
        rag_uid_int = int(rag_unit_id_val) if rag_unit_id_val is not None else 0
    except (TypeError, ValueError):
        rag_uid_int = 0
    if rag_uid_int > 0:
        unit_sel = (
            supabase.table("Rag_Unit")
            .select("unit_name, unit_type")
            .eq("rag_unit_id", rag_uid_int)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if unit_sel.data:
            u0 = unit_sel.data[0]
            if not grade_unit_filter:
                grade_unit_filter = (u0.get("unit_name") or "").strip() or None
            try:
                exam_grade_unit_type = int(u0.get("unit_type") or 0)
            except (TypeError, ValueError):
                exam_grade_unit_type = 0

    try:
        rag_id = int(rag_id_from_setting)
        stem, rag_zip_tab_id = get_rag_stem_from_rag_id(supabase, rag_id, unit_name=grade_unit_filter)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

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
            _cleanup_grade_workspace(work_dir)
            return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
    except Exception as e:
        _cleanup_grade_workspace(work_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            rag_zip_path.unlink(missing_ok=True)
        except Exception:
            pass

    job_id = str(uuid.uuid4())
    _exam_grade_job_results[job_id] = {"status": "pending", "result": None, "error": None}
    exam_quiz_id_int = int(body.exam_quiz_id)
    insert_fn = lambda rd, qa: _update_exam_quiz_with_grade(rd, qa, exam_quiz_id=exam_quiz_id_int)
    background_tasks.add_task(
        _run_grade_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        body.quiz_answer or "",
        _exam_grade_job_results,
        insert_fn,
        (body.answer_user_prompt_text or "").strip(),
        exam_quiz_id=exam_quiz_id_int,
        unit_type=exam_grade_unit_type,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id})


# --- GET /exam/tab/quiz/grade-result/{job_id} ---

@router.get("/tab/quiz/grade-result/{job_id}", tags=["exam"])
async def get_exam_grade_result(job_id: str, _person_id: PersonId):
    """
    輪詢 Exam 評分結果（搭配 `POST /exam/tab/quiz/llm-grade`）。
    status: pending | ready | error；
    ready 時 result 含 quiz_grade、quiz_comments、exam_quiz_id（已寫入 Exam_Quiz）。
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


# --- POST /exam/tab/quiz/rate（置於本模組末尾）---

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
