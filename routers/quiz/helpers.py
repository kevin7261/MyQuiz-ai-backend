"""routers.quiz（測驗／Test）helpers。

定位等同 exam 之於 rag：Quiz 是搭配 bank 的「應試／測驗」層。
- Quiz（測驗）→ Quiz_Group（自既有 Bank_Group 快照之題組）→ Quiz_QA（逐題出題／批改，無追問）。
- 出題／批改沿用 **bank 的 LLM 管線**（services.bank_generation／services.bank_answering）與 bank 的內容
  （RAG ZIP／逐字稿）；金鑰／模型走 quiz- 設定（utils.quiz_llm_key）。
- 程式不與 exam／rag 共用（不 import routers.exam／routers.zip／services.quiz_generation／services.answering）。

對應 bank 題組的設計：
- question_system_prompt_text → 連續出題規定（織入出題 system prompt，最高優先）
- question_user_prompt_text   → 出題 user prompt
- 同題組既有題目的題幹        → 已出過題目（連續出題勿重複）
- answer_user_prompt_text     → 批改 user prompt
- 以 qa_count 為逐題產生之上限；無「追問」概念。
"""

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response

from services.bank_answering import (
    answer_critique_plain_text_from_result,
    cleanup_answer_workspace,
    run_bank_answer_job_background,
)
from services.quiz_asking import (
    format_ask_history_body,
    format_quiz_qa_history_body,
    run_quiz_ask_job,
    run_quiz_ask_transcript_only,
)
from utils.llm_error import format_llm_error, is_llm_call_error, llm_error_json_response
from utils.bank_course import (
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.bank_storage import get_zip_path
from utils.bank_stem import transcript_from_row
from utils.qa_count import normalize_qa_count
from utils.quiz_llm_key import get_quiz_api_key, get_quiz_llm_model
from utils.supabase import get_supabase
from utils.serialization import to_json_safe
from utils.taipei_time import now_taipei_iso
from utils.fs import safe_unlink

# 出題／批改之單元內容存取與題目產生 glue（bank 與 quiz 共用，見 routers.bank.question_content）
from routers.bank.question_content import (
    TRANSCRIPT_UNIT_TYPES,
    BankAnswerSetupError,
    fetch_bank_unit_for_llm,
    generate_question_fields_from_bank_unit,
    prepare_bank_answer_workspace,
    rag_zip_page_id_from_unit,
)

_logger = logging.getLogger("routers.quiz")

# 記憶體批改結果（鍵為 job_id）；供 GET /quiz/qa/answer-result/{job_id} 輪詢。
_quiz_answer_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Quiz 三表查詢（新表，course_id 必有，直接篩選）
# ---------------------------------------------------------------------------


def fetch_quiz_page_row(supabase, quiz_page_id: str, course_id: int, *, cols: str = "*") -> dict | None:
    """依 quiz_page_id 取未刪除 Quiz 一列。"""
    sel = (
        supabase.table("Quiz")
        .select(cols)
        .eq("quiz_page_id", quiz_page_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    return sel.data[0] if sel.data else None


def fetch_quiz_group_row(supabase, quiz_group_id: int, course_id: int, *, cols: str = "*") -> dict | None:
    """依 quiz_group_id 取未刪除 Quiz_Group 一列。"""
    sel = (
        supabase.table("Quiz_Group")
        .select(cols)
        .eq("quiz_group_id", quiz_group_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    return sel.data[0] if sel.data else None


def require_quiz_group_owner(
    supabase,
    quiz_group_id: int,
    course_id: int,
    caller_person_id: str,
    *,
    cols: str = "*",
    forbidden_detail: str = "無權存取該 Quiz_Group",
) -> dict:
    """取 Quiz_Group 並驗證擁有者；不存在回 404、非擁有者回 403（detail 依呼叫情境傳入）。"""
    group = fetch_quiz_group_row(supabase, quiz_group_id, course_id, cols=cols)
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 quiz_group_id={quiz_group_id} 的 Quiz_Group，或已刪除")
    if (group.get("person_id") or "").strip() != caller_person_id:
        raise HTTPException(status_code=403, detail=forbidden_detail)
    return group


def fetch_quiz_qa_row(supabase, quiz_qa_id: int, course_id: int, *, cols: str = "*") -> dict | None:
    """依 quiz_qa_id 取未刪除 Quiz_QA 一列。"""
    sel = (
        supabase.table("Quiz_QA")
        .select(cols)
        .eq("quiz_qa_id", quiz_qa_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    return sel.data[0] if sel.data else None


def fetch_quiz_ask_row(supabase, quiz_ask_id: int, course_id: int, *, cols: str = "*") -> dict | None:
    """依 quiz_ask_id 取未刪除 Quiz_Ask 一列。"""
    sel = (
        supabase.table("Quiz_Ask")
        .select(cols)
        .eq("quiz_ask_id", quiz_ask_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    return sel.data[0] if sel.data else None


def quiz_ask_rows_for_group(supabase, quiz_group_id: int, course_id: int, *, cols: str = "*") -> list[dict]:
    """依 quiz_group_id 取所有未刪除 Quiz_Ask，依 created_at 升序（由舊到新）。"""
    sel = (
        supabase.table("Quiz_Ask")
        .select(cols)
        .eq("quiz_group_id", quiz_group_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .order("created_at", desc=False)
        .execute()
    )
    return sel.data or []


def quiz_qa_rows_for_group(supabase, quiz_group_id: int, course_id: int, *, cols: str = "*") -> list[dict]:
    """依 quiz_group_id 取所有未刪除 Quiz_QA，依 question_series_index、created_at 升序。"""
    sel = (
        supabase.table("Quiz_QA")
        .select(cols)
        .eq("quiz_group_id", quiz_group_id)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .order("question_series_index", desc=False)
        .order("created_at", desc=False)
        .execute()
    )
    return sel.data or []


def renumber_quiz_qa_indices(supabase, quiz_group_id: int, course_id: int) -> None:
    """刪題後將同題組剩餘 Quiz_QA 的 question_series_index 重排為 1, 2, 3, …。"""
    rows = quiz_qa_rows_for_group(
        supabase, quiz_group_id, course_id, cols="quiz_qa_id, question_series_index"
    )
    ts = now_taipei_iso()
    for idx, row in enumerate(rows, start=1):
        qa_id = row.get("quiz_qa_id")
        if qa_id is None:
            continue
        try:
            current = int(row.get("question_series_index") or 0)
        except (TypeError, ValueError):
            current = 0
        if current == idx:
            continue
        supabase.table("Quiz_QA").update(
            {"question_series_index": idx, "updated_at": ts}
        ).eq("quiz_qa_id", qa_id).eq("deleted", False).execute()


# ---------------------------------------------------------------------------
# Bank 讀取（建立快照／出題內容；bank 表可能缺 course_id 欄位，沿用 bank fallback）
# ---------------------------------------------------------------------------


def fetch_bank_group_for_snapshot(supabase, bank_group_id: int, course_id: int) -> dict | None:
    """讀來源 Bank_Group（未刪除）整列，供快照成 Quiz_Group。"""

    def build(with_course_filter: bool):
        q = (
            supabase.table("Bank_Group")
            .select("*")
            .eq("bank_group_id", bank_group_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    sel = execute_with_course_id_fallback("Bank_Group", build, course_id)
    return sel.data[0] if sel.data else None


# fetch_bank_unit_for_llm／rag_zip_page_id_from_unit 自 routers.bank.question_content 匯入（bank 與 quiz 共用）。


def list_bank_groups_for_quiz(course_id: int) -> list[dict]:
    """列出可選用的 Bank_Group（for_exam=true、未刪除），附其題庫 tab_name 與單元 unit_name／unit_type。"""
    supabase = get_supabase()

    def build(with_course_filter: bool):
        q = (
            supabase.table("Bank_Group")
            .select("*")
            .eq("for_exam", True)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("created_at", desc=False)

    groups = execute_with_course_id_fallback("Bank_Group", build, course_id).data or []

    # 補上各題組所屬單元的 unit_name／unit_type（供前端顯示）
    unit_ids = list(dict.fromkeys(
        int(g.get("bank_unit_id")) for g in groups if g.get("bank_unit_id") is not None
    ))
    units_by_id: dict[int, dict] = {}
    if unit_ids:
        def build_units(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Bank_Unit", "bank_unit_id, unit_name, unit_type, course_id", with_course_filter
            )
            q = (
                supabase.table("Bank_Unit")
                .select(cols)
                .in_("bank_unit_id", unit_ids)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q

        for u in execute_with_course_id_fallback("Bank_Unit", build_units, course_id).data or []:
            uid = u.get("bank_unit_id")
            if uid is not None:
                units_by_id[int(uid)] = u

    # 補上各題組所屬題庫的 tab_name（題庫名，供前端顯示「題庫 › 單元 › 題組」）
    page_ids = list(dict.fromkeys(
        (g.get("bank_page_id") or "").strip() for g in groups if (g.get("bank_page_id") or "").strip()
    ))
    tab_name_by_page_id: dict[str, str] = {}
    if page_ids:
        def build_banks(with_course_filter: bool):
            q = (
                supabase.table("Bank")
                .select("bank_page_id, tab_name, course_id")
                .in_("bank_page_id", page_ids)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q

        for b in execute_with_course_id_fallback("Bank", build_banks, course_id).data or []:
            pid = (b.get("bank_page_id") or "").strip()
            if pid:
                tab_name_by_page_id[pid] = (b.get("tab_name") or "").strip()

    for g in groups:
        uid = g.get("bank_unit_id")
        unit = units_by_id.get(int(uid)) if uid is not None else None
        g["unit_name"] = (unit or {}).get("unit_name") or ""
        g["unit_type"] = (unit or {}).get("unit_type") or 0
        g["tab_name"] = tab_name_by_page_id.get((g.get("bank_page_id") or "").strip(), "")
    return groups


# ---------------------------------------------------------------------------
# 建立 Quiz_Group（自 Bank_Group 快照）
# ---------------------------------------------------------------------------


def build_quiz_group_snapshot_row(
    *,
    quiz_page_id: str,
    person_id: str,
    course_id: int,
    bank_group: dict,
    bank_unit: dict | None,
    group_name_override: str = "",
) -> dict[str, Any]:
    """把 Bank_Group（＋其 Bank_Unit）快照成一筆 Quiz_Group row（凍結 prompt／模型／qa_count／單元資訊）。"""
    ts = now_taipei_iso()
    unit = bank_unit or {}
    group_name = (group_name_override or "").strip() or (bank_group.get("group_name") or "").strip()
    return {
        "quiz_page_id": quiz_page_id,
        "bank_page_id": (bank_group.get("bank_page_id") or unit.get("bank_page_id") or "").strip(),
        "bank_unit_id": int(bank_group.get("bank_unit_id") or 0),
        "bank_group_id": int(bank_group.get("bank_group_id") or 0),
        "person_id": person_id,
        "course_id": course_id,
        "unit_name": (unit.get("unit_name") or "").strip(),
        "unit_type": int(unit.get("unit_type") or 0),
        "group_name": group_name,
        "question_system_prompt_text": bank_group.get("question_system_prompt_text") or "",
        "question_user_prompt_text": bank_group.get("question_user_prompt_text") or "",
        "qa_count": normalize_qa_count(bank_group.get("qa_count")),
        "question_llm_model": (bank_group.get("question_llm_model") or "").strip(),
        "answer_user_prompt_text": bank_group.get("answer_user_prompt_text") or "",
        "answer_llm_model": (bank_group.get("answer_llm_model") or "").strip(),
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }


# ---------------------------------------------------------------------------
# GET /quiz/pages 巢狀（題組與題目掛在測驗下）
# ---------------------------------------------------------------------------


def _qas_by_quiz_group_ids(group_ids: list[int], course_id: int) -> dict[int, list[dict]]:
    """依 quiz_group_id 取未刪除 Quiz_QA，回傳 quiz_group_id -> qas[]（依 question_series_index 升序）。"""
    if not group_ids:
        return {}
    supabase = get_supabase()
    resp = (
        supabase.table("Quiz_QA")
        .select("*")
        .in_("quiz_group_id", group_ids)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .order("question_series_index", desc=False)
        .order("created_at", desc=False)
        .execute()
    )
    out: dict[int, list[dict]] = {gid: [] for gid in group_ids}
    for row in resp.data or []:
        gid = row.get("quiz_group_id")
        if gid is not None:
            out.setdefault(int(gid), []).append(row)
    return out


def groups_by_quiz_page_ids(page_ids: list[str], course_id: int) -> dict[str, list[dict]]:
    """依 quiz_page_id 取未刪除 Quiz_Group（含其 Quiz_QA），回傳 quiz_page_id -> groups[]。"""
    if not page_ids:
        return {}
    supabase = get_supabase()
    resp = (
        supabase.table("Quiz_Group")
        .select("*")
        .in_("quiz_page_id", page_ids)
        .eq("course_id", course_id)
        .eq("deleted", False)
        .order("created_at", desc=False)
        .execute()
    )
    groups = resp.data or []

    group_ids = list(dict.fromkeys(
        int(g.get("quiz_group_id")) for g in groups if g.get("quiz_group_id") is not None
    ))
    qas_by_group = _qas_by_quiz_group_ids(group_ids, course_id)

    out: dict[str, list[dict]] = {pid: [] for pid in page_ids}
    for g in groups:
        gid = g.get("quiz_group_id")
        g["qas"] = qas_by_group.get(int(gid), []) if gid is not None else []
        pid = g.get("quiz_page_id")
        if pid is not None:
            out.setdefault(str(pid), []).append(g)
    return out


# ---------------------------------------------------------------------------
# 逐題出題（LLM；同步，沿用 bank 出題管線）
# ---------------------------------------------------------------------------


def _resolve_quiz_llm_params(group: dict, course_id: int) -> tuple[str, str, str, str]:
    """解析出題所需 api_key／llm_model／qup／qsp（不呼叫 LLM）；prompt 一律取自 Quiz_Group。"""
    api_key = get_quiz_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 Quiz API Key：PUT /v1/quiz/llm-api-key（Course_Setting key=quiz-api-key，依 course_id）",
        )
    llm_model = (group.get("question_llm_model") or "").strip() or get_quiz_llm_model(course_id)
    qup = (group.get("question_user_prompt_text") or "").strip()
    qsp = (group.get("question_system_prompt_text") or "").strip()
    return api_key, llm_model, qup, qsp


# 出題 glue 已抽至 routers.bank.question_content.generate_question_fields_from_bank_unit（bank 與 quiz 共用）。


def quiz_llm_generate_qa_impl(
    *,
    quiz_group_id: int,
    caller_person_id: str,
    course_id: int,
):
    """在題組內產生下一題：依題組 prompt 與既有題目（勿重複）出一題，寫入 Quiz_QA。"""
    supabase = get_supabase()
    group = require_quiz_group_owner(
        supabase, quiz_group_id, course_id, caller_person_id, forbidden_detail="無權於該題組出題"
    )
    pid = (group.get("person_id") or "").strip()

    qa_count = normalize_qa_count(group.get("qa_count"))

    existing = quiz_qa_rows_for_group(
        supabase, quiz_group_id, course_id, cols="quiz_qa_id, question_series_index, question_content"
    )
    if len(existing) >= qa_count:
        raise HTTPException(
            status_code=409,
            detail=f"本題組已達 qa_count 上限（{qa_count} 題），無法再出題",
        )

    api_key, llm_model, qup, qsp = _resolve_quiz_llm_params(group, course_id)

    # 既有題幹 → 已出過題目（連續出題、勿重複）
    prior_items = [
        {"quiz_content": (q.get("question_content") or "").strip()}
        for q in existing
        if (q.get("question_content") or "").strip()
    ]
    ask_rows = quiz_ask_rows_for_group(
        supabase,
        quiz_group_id,
        course_id,
        cols="ask_user_prompt_text, answer_content, created_at",
    )
    ask_history_body = format_ask_history_body(ask_rows)

    try:
        fields = generate_question_fields_from_bank_unit(
            supabase,
            bank_unit_id=int(group.get("bank_unit_id") or 0),
            bank_page_id_fallback=(group.get("bank_page_id") or ""),
            course_id=course_id,
            api_key=api_key,
            llm_model=llm_model,
            qup=qup,
            qsp=qsp,
            prior_items=prior_items,
            ask_history_body=ask_history_body,
        )
        series_index = len(existing) + 1
        ts = now_taipei_iso()
        # prompt 為出題當下自題組複製、凍結（question_*、answer_user_prompt_text）。
        # question_llm_model 記本次出題所用；answer_llm_model 於批改完成時寫入。
        qa_row: dict[str, Any] = {
            "quiz_group_id": quiz_group_id,
            "quiz_page_id": (group.get("quiz_page_id") or "").strip(),
            "bank_page_id": fields["bank_page_id"],
            "bank_unit_id": fields["bank_unit_id"],
            "bank_group_id": int(group.get("bank_group_id") or 0),
            "person_id": pid,
            "course_id": course_id,
            "question_series_index": series_index,
            "question_system_prompt_text": qsp,
            "question_user_prompt_text": qup,
            "question_content": fields["question_content"],
            "question_hint": fields["question_hint"],
            "question_answer_reference": fields["question_answer_reference"],
            "question_reason": fields["question_reason"],
            "question_rate": 0,
            "question_llm_model": llm_model,
            "answer_user_prompt_text": (group.get("answer_user_prompt_text") or ""),
            "answer_content": "",
            "answer_critique": None,
            "answer_rate": 0,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = supabase.table("Quiz_QA").insert(qa_row).execute()
        if not ins.data:
            raise HTTPException(status_code=500, detail="寫入 Quiz_QA 失敗（無回傳資料）")
        created = ins.data[0]
        out = {
            "question_llm_model": llm_model,
            "qa_count": qa_count,
            "generated_count": series_index,
            **to_json_safe(created),
        }
        return Response(
            content=json.dumps(out, ensure_ascii=False).encode("utf-8"),
            media_type="application/json; charset=utf-8",
        )
    except ValueError:
        _logger.exception("quiz_llm_generate_qa_impl 參數錯誤 quiz_group_id=%s", quiz_group_id)
        raise HTTPException(status_code=400, detail="參數錯誤，請稍後再試")
    except HTTPException:
        raise
    except Exception as e:
        if is_llm_call_error(e):
            return llm_error_json_response(
                {
                    "llm_error": format_llm_error(e),
                    "quiz_group_id": quiz_group_id,
                    "question_content": "",
                    "question_hint": "",
                    "question_answer_reference": "",
                    "question_llm_model": llm_model,
                }
            )
        _logger.exception("quiz_llm_generate_qa_impl 失敗 quiz_group_id=%s", quiz_group_id)
        raise HTTPException(status_code=500, detail="操作失敗，請稍後再試")


def quiz_llm_regenerate_qa_impl(
    *,
    quiz_qa_id: int,
    caller_person_id: str,
    course_id: int,
):
    """重新產生**同一題**（原地覆寫同一 quiz_qa_id）。

    不刪除、不新增任何 Quiz_QA，亦不改動 question_series_index；只把這一題的 question_* 內容用 LLM
    重新產生並 UPDATE 回原列。「勿重複」清單為題組內**此題之前**的題（排除本題），重出後本題舊作答／批改清空。
    不檢查 qa_count 上限（重出佔用既有名額，非新增）。
    """
    supabase = get_supabase()
    qa = fetch_quiz_qa_row(supabase, quiz_qa_id, course_id, cols="*")
    if not qa:
        raise HTTPException(status_code=404, detail=f"找不到 quiz_qa_id={quiz_qa_id} 的 Quiz_QA，或已刪除")
    quiz_group_id = int(qa.get("quiz_group_id") or 0)
    if quiz_group_id <= 0:
        raise HTTPException(status_code=400, detail="該題的 quiz_group_id 無效")
    group = require_quiz_group_owner(
        supabase, quiz_group_id, course_id, caller_person_id, forbidden_detail="無權重出該題"
    )

    existing = quiz_qa_rows_for_group(
        supabase, quiz_group_id, course_id, cols="quiz_qa_id, question_series_index, question_content"
    )
    try:
        current_series_index = int(qa.get("question_series_index") or 0)
    except (TypeError, ValueError):
        current_series_index = 0
    # 勿重複：只取「此題之前」的題（question_series_index 較小者），與逐題出題一致。
    prior_items = [
        {"quiz_content": (q.get("question_content") or "").strip()}
        for q in existing
        if int(q.get("quiz_qa_id") or 0) != quiz_qa_id
        and (current_series_index <= 0 or int(q.get("question_series_index") or 0) < current_series_index)
        and (q.get("question_content") or "").strip()
    ]
    api_key, llm_model, qup, qsp = _resolve_quiz_llm_params(group, course_id)
    ask_rows = quiz_ask_rows_for_group(
        supabase,
        quiz_group_id,
        course_id,
        cols="ask_user_prompt_text, answer_content, created_at",
    )
    ask_history_body = format_ask_history_body(ask_rows)

    try:
        fields = generate_question_fields_from_bank_unit(
            supabase,
            bank_unit_id=int(group.get("bank_unit_id") or 0),
            bank_page_id_fallback=(group.get("bank_page_id") or ""),
            course_id=course_id,
            api_key=api_key,
            llm_model=llm_model,
            qup=qup,
            qsp=qsp,
            prior_items=prior_items,
            ask_history_body=ask_history_body,
        )
        ts = now_taipei_iso()
        update_row: dict[str, Any] = {
            "question_system_prompt_text": qsp,
            "question_user_prompt_text": qup,
            "question_content": fields["question_content"],
            "question_hint": fields["question_hint"],
            "question_answer_reference": fields["question_answer_reference"],
            "question_reason": fields["question_reason"],
            "question_llm_model": llm_model,
            # 重出後舊作答／批改／評分失效，一併清空（question_series_index 不變）；
            # 批改規則紀錄重置為題組現值（同新出一題），待下次批改完成再寫入實際使用值
            "answer_content": "",
            "answer_critique": None,
            "answer_llm_model": None,
            "answer_user_prompt_text": (group.get("answer_user_prompt_text") or ""),
            "answer_rate": 0,
            "updated_at": ts,
        }
        supabase.table("Quiz_QA").update(update_row).eq("quiz_qa_id", quiz_qa_id).eq("deleted", False).execute()
        updated = fetch_quiz_qa_row(supabase, quiz_qa_id, course_id, cols="*") or {**qa, **update_row}
        qa_count = normalize_qa_count(group.get("qa_count"))
        out = {
            "question_llm_model": llm_model,
            "qa_count": qa_count,
            "generated_count": int(
                updated.get("question_series_index") or qa.get("question_series_index") or 0
            ),
            **to_json_safe(updated),
        }
        return Response(
            content=json.dumps(out, ensure_ascii=False).encode("utf-8"),
            media_type="application/json; charset=utf-8",
        )
    except ValueError:
        _logger.exception("quiz_llm_regenerate_qa_impl 參數錯誤 quiz_qa_id=%s", quiz_qa_id)
        raise HTTPException(status_code=400, detail="參數錯誤，請稍後再試")
    except HTTPException:
        raise
    except Exception as e:
        if is_llm_call_error(e):
            return llm_error_json_response(
                {
                    "llm_error": format_llm_error(e),
                    "quiz_group_id": quiz_group_id,
                    "quiz_qa_id": quiz_qa_id,
                    "question_content": "",
                    "question_hint": "",
                    "question_answer_reference": "",
                    "question_llm_model": llm_model,
                }
            )
        _logger.exception("quiz_llm_regenerate_qa_impl 失敗 quiz_qa_id=%s", quiz_qa_id)
        raise HTTPException(status_code=500, detail="操作失敗，請稍後再試")


# ---------------------------------------------------------------------------
# 批改（LLM；非同步，沿用 bank 批改管線）
# ---------------------------------------------------------------------------


def update_quiz_qa_with_answer(
    result_dict: dict,
    answer_content: str,
    *,
    quiz_qa_id: int,
    answer_llm_model: str = "",
    answer_user_prompt_text: str = "",
) -> tuple[str, int] | None:
    """背景批改完成後更新 Quiz_QA：寫 answer_content、answer_critique，並把本次批改**實際使用**的
    answer_llm_model／answer_user_prompt_text 寫回（QA 列記「這一題各次呼叫實際用了什麼」：
    出題規則於出題時寫入、批改規則於批改時寫入）；question_* 為出題時凍結，不覆寫。回傳 (id_key, id_val)。"""
    if quiz_qa_id <= 0:
        return None
    critique_text = answer_critique_plain_text_from_result(result_dict)
    ts = now_taipei_iso()
    row: dict[str, Any] = {
        "answer_content": answer_content or "",
        "answer_critique": critique_text,
        "updated_at": ts,
    }
    if (answer_llm_model or "").strip():
        row["answer_llm_model"] = answer_llm_model.strip()
    if (answer_user_prompt_text or "").strip():
        row["answer_user_prompt_text"] = answer_user_prompt_text.strip()
    try:
        supabase = get_supabase()
        supabase.table("Quiz_QA").update(row).eq("quiz_qa_id", quiz_qa_id).eq("deleted", False).execute()
        chk = (
            supabase.table("Quiz_QA")
            .select("answer_critique")
            .eq("quiz_qa_id", quiz_qa_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not chk.data:
            _logger.warning("Quiz_QA update 後讀不到列（quiz_qa_id=%s 不存在、已刪除或遭 RLS 擋）", quiz_qa_id)
            return None
        if chk.data[0].get("answer_critique") is None:
            _logger.warning("Quiz_QA 讀回 answer_critique 為空（quiz_qa_id=%s）", quiz_qa_id)
            return None
        return ("quiz_qa_id", quiz_qa_id)
    except Exception as e:
        _logger.warning("Quiz_QA answer update 失敗: %s", e, exc_info=True)
    return None


async def enqueue_quiz_qa_answer_job(
    background_tasks: BackgroundTasks,
    caller_person_id: str,
    course_id: int,
    *,
    quiz_qa_id: int,
    answer_content: str,
) -> JSONResponse:
    """將 Quiz_QA 批改排入 BackgroundTasks；回傳 202 + job_id（輪詢 GET /quiz/qa/answer-result/{job_id}）。"""
    if quiz_qa_id <= 0:
        return JSONResponse(status_code=400, content={"error": "quiz_qa_id 必填且須為大於 0 的整數"})

    supabase = get_supabase()
    qa = fetch_quiz_qa_row(
        supabase,
        quiz_qa_id,
        course_id,
        cols=(
            "quiz_qa_id, quiz_group_id, bank_unit_id, person_id, question_content, "
            "question_user_prompt_text, answer_user_prompt_text, answer_llm_model"
        ),
    )
    if not qa:
        return JSONResponse(status_code=404, content={"error": f"找不到 quiz_qa_id={quiz_qa_id} 的 Quiz_QA"})
    pid = (qa.get("person_id") or "").strip()
    if pid != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權批改該 Quiz_QA"})

    quiz_content = (qa.get("question_content") or "").strip()
    if not quiz_content:
        return JSONResponse(
            status_code=400,
            content={"error": "該 Quiz_QA 尚無題幹（question_content 為空），請先出題後再批改"},
        )

    # 批改設定一律用「題組現值」：規則／模型每次批改時自 Quiz_Group 重抓（保證最新）；
    # 題組規則為空才回退此題出題當下凍結的快照（舊資料相容）。
    # 模型不看 QA 凍結值：題組 answer_llm_model 空 → 課程 quiz-llm-model 現值；
    # 批改完成後由 update_quiz_qa_with_answer 把實際使用的模型／批改規則寫回
    # Quiz_QA.answer_llm_model／answer_user_prompt_text（QA 列記各次呼叫實際使用值）。
    quiz_group_id = int(qa.get("quiz_group_id") or 0)
    group = (
        fetch_quiz_group_row(
            supabase,
            quiz_group_id,
            course_id,
            cols="quiz_group_id, question_user_prompt_text, answer_user_prompt_text, answer_llm_model, course_id",
        )
        or {}
    )
    quiz_user_prompt = (
        (group.get("question_user_prompt_text") or "").strip()
        or (qa.get("question_user_prompt_text") or "").strip()
    )
    answer_user_prompt = (
        (group.get("answer_user_prompt_text") or "").strip()
        or (qa.get("answer_user_prompt_text") or "").strip()
    )

    api_key = get_quiz_api_key(course_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "請設定 Quiz API Key：PUT /v1/quiz/llm-api-key（Course_Setting key=quiz-api-key，依 course_id）"},
        )
    llm_model = (group.get("answer_llm_model") or "").strip() or get_quiz_llm_model(course_id)

    bank_unit_id = int(qa.get("bank_unit_id") or 0)
    try:
        unit_type, transcript_answer, work_dir = prepare_bank_answer_workspace(
            supabase, bank_unit_id=bank_unit_id, course_id=course_id, prefix="myquizai_quiz_answer"
        )
    except BankAnswerSetupError as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    job_id = str(uuid.uuid4())
    _quiz_answer_job_results[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "llm_error": None,
        "quiz_qa_id": quiz_qa_id,
    }

    def insert_fn(rd, ans):
        return update_quiz_qa_with_answer(
            rd,
            ans,
            quiz_qa_id=quiz_qa_id,
            answer_llm_model=llm_model,
            answer_user_prompt_text=answer_user_prompt,
        )

    background_tasks.add_task(
        run_bank_answer_job_background,
        job_id,
        work_dir,
        api_key,
        quiz_content,
        answer_content or "",
        _quiz_answer_job_results,
        insert_fn,
        answer_user_prompt_text=answer_user_prompt,
        bank_qa_id=quiz_qa_id,
        unit_type=unit_type,
        transcript_answer=transcript_answer,
        quiz_user_prompt_text=quiz_user_prompt,
        llm_model=llm_model,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "answer_llm_model": llm_model})


# ---------------------------------------------------------------------------
# 追問（LLM；同步）：對題組對應之 Bank 課程內容發問，寫入 Quiz_Ask
# ---------------------------------------------------------------------------


def quiz_llm_ask_impl(
    *,
    quiz_group_id: int,
    ask_user_prompt_text: str = "",
    caller_person_id: str,
    course_id: int,
):
    """
    學生於題組出題後，對該題組對應 **Bank 單元的課程內容**發問（POST /quiz/groups/{id}/llm-ask）。
    依課程內容（逐字稿／向量檢索）、本題組測驗紀錄與先前追問紀錄同步請 LLM 回答，並於 public.Quiz_Ask 新增一列。
    回傳該列（含 answer_content）。LLM 失敗回 200 + llm_error。
    """
    ask_text = (ask_user_prompt_text or "").strip()
    if not ask_text:
        raise HTTPException(status_code=400, detail="ask_user_prompt_text 不可為空白")

    supabase = get_supabase()
    group = require_quiz_group_owner(
        supabase, quiz_group_id, course_id, caller_person_id, forbidden_detail="無權對該 Quiz_Group 發問"
    )
    pid = (group.get("person_id") or "").strip()

    api_key = get_quiz_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 Quiz API Key：PUT /v1/quiz/llm-api-key（Course_Setting key=quiz-api-key，依 course_id）",
        )
    llm_model = get_quiz_llm_model(course_id)

    bank_unit_id = int(group.get("bank_unit_id") or 0)
    if bank_unit_id <= 0:
        raise HTTPException(status_code=400, detail="該題組對應的 bank_unit_id 無效")
    unit = fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"找不到 bank_unit_id={bank_unit_id} 的 Bank_Unit")

    try:
        unit_type_val = int(unit.get("unit_type") or 0)
    except (TypeError, ValueError):
        unit_type_val = 0
    transcript_text = transcript_from_row(unit)
    question_user_prompt = (group.get("question_user_prompt_text") or "").strip()
    bank_group_id = int(group.get("bank_group_id") or 0)

    qa_rows = quiz_qa_rows_for_group(
        supabase,
        quiz_group_id,
        course_id,
        cols=(
            "question_series_index, question_content, question_hint, "
            "question_answer_reference, answer_content, answer_critique"
        ),
    )
    prior_asks = quiz_ask_rows_for_group(
        supabase,
        quiz_group_id,
        course_id,
        cols="ask_user_prompt_text, answer_content, created_at",
    )
    quiz_qa_history_body = format_quiz_qa_history_body(qa_rows)
    ask_history_body = format_ask_history_body(prior_asks)

    work_dir: Path | None = None
    try:
        if unit_type_val in TRANSCRIPT_UNIT_TYPES:
            if not transcript_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="回答用 transcript 未設定：請於 Bank_Unit 設定 transcript（單元 2／3／4）",
                )
            answer_text = run_quiz_ask_transcript_only(
                api_key,
                transcript_text,
                ask_text,
                question_user_prompt_text=question_user_prompt,
                quiz_qa_history_body=quiz_qa_history_body,
                ask_history_body=ask_history_body,
                quiz_group_id=quiz_group_id,
                bank_group_id=bank_group_id if bank_group_id > 0 else None,
                llm_model=llm_model,
            )
        else:
            rag_zip_page_id = rag_zip_page_id_from_unit(unit)
            rag_zip_path = get_zip_path(rag_zip_page_id) if rag_zip_page_id else None
            if not rag_zip_path or not rag_zip_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP（page_id={rag_zip_page_id}），請確認該單元已 build-zip",
                )
            work_dir = Path(tempfile.mkdtemp(prefix="myquizai_quiz_ask_"))
            try:
                shutil.copy(rag_zip_path, work_dir / "ref.zip")
            finally:
                safe_unlink(rag_zip_path)
            answer_text = run_quiz_ask_job(
                work_dir,
                api_key,
                ask_text,
                question_user_prompt_text=question_user_prompt,
                quiz_qa_history_body=quiz_qa_history_body,
                ask_history_body=ask_history_body,
                quiz_group_id=quiz_group_id,
                bank_group_id=bank_group_id if bank_group_id > 0 else None,
                unit_type=unit_type_val,
                llm_model=llm_model,
            )
    except HTTPException:
        raise
    except ValueError:
        _logger.exception("POST /quiz/groups/{id}/llm-ask 參數錯誤 quiz_group_id=%s", quiz_group_id)
        raise HTTPException(status_code=400, detail="參數錯誤，請稍後再試")
    except Exception as e:
        if is_llm_call_error(e):
            return llm_error_json_response({
                "llm_error": format_llm_error(e),
                "quiz_group_id": quiz_group_id,
                "answer_content": "",
            })
        _logger.exception("POST /quiz/groups/{id}/llm-ask 回答失敗 quiz_group_id=%s", quiz_group_id)
        raise HTTPException(status_code=500, detail="操作失敗，請稍後再試") from e
    finally:
        if work_dir is not None:
            cleanup_answer_workspace(work_dir)

    ts = now_taipei_iso()
    # ask_llm_model 記本次追問回答所用（同步完成後與 answer_content 一併寫入，對齊 Quiz_QA.question_llm_model）。
    ask_row: dict[str, Any] = {
        "quiz_group_id": quiz_group_id,
        "quiz_page_id": (group.get("quiz_page_id") or "").strip(),
        "bank_page_id": (group.get("bank_page_id") or "").strip(),
        "bank_unit_id": bank_unit_id,
        "bank_group_id": bank_group_id,
        "person_id": pid,
        "course_id": course_id,
        "unit_name": (group.get("unit_name") or "").strip(),
        "unit_type": unit_type_val,
        "group_name": (group.get("group_name") or "").strip(),
        "ask_user_prompt_text": ask_text,
        "ask_llm_model": llm_model,
        "answer_content": answer_text,
        "answer_rate": 0,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }
    try:
        ins = supabase.table("Quiz_Ask").insert(ask_row).execute()
    except Exception as e:
        _logger.exception("POST /quiz/groups/{id}/llm-ask 寫入 Quiz_Ask 失敗 quiz_group_id=%s", quiz_group_id)
        raise HTTPException(status_code=500, detail="寫入 Quiz_Ask 失敗，請稍後再試") from e
    if not ins.data:
        raise HTTPException(status_code=500, detail="寫入 Quiz_Ask 失敗（無回傳資料）")
    return to_json_safe(ins.data[0])
