"""routers.bank 題組（Bank_Group）／問答（Bank_QA）helpers。

出題／批改的 LLM prompt 與管線使用 **bank 專屬** 的 services.bank_generation 與 services.bank_answering，
與 rag／exam 完全無關（可自由修改不影響其他模組）；所有資料表操作（Bank_Group／Bank_QA）亦為 bank 專屬。

對應 rag 的設計：
- question_system_prompt_text → quiz_system_prompt_text（連續出題規定，織入出題 system prompt，最高優先）
- question_user_prompt_text   → quiz_user_prompt_text（出題 user prompt）
- 同題組既有題目的題幹        → quiz_history_list_prompt_text（已出過題目，連續出題勿重複）
- answer_user_prompt_text     → 批改 user prompt
- 無「追問」概念；以 qa_count 為逐題產生之上限。
"""

import json
import logging
import uuid
from typing import Any

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response
from services.bank_answering import (
    answer_critique_plain_text_from_result,
    run_bank_answer_job_background,
)
from utils.bank_llm_error import format_llm_error, is_llm_call_error, llm_error_json_response
from utils.bank_llm_key import get_bank_api_key, get_bank_llm_model
from utils.qa_count import normalize_qa_count
from utils.taipei_time import now_taipei_iso
from utils.bank_course import (
    execute_with_course_id_fallback,
    insert_bank_child_row,
    select_without_course_id_if_needed,
)
from utils.supabase import get_supabase
from utils.serialization import to_json_safe

# 出題／批改之單元內容存取與題目產生 glue（bank 與 quiz 共用，見 routers.bank.question_content）
from .question_content import (
    BankAnswerSetupError,
    fetch_bank_unit_for_llm,
    generate_question_fields_from_bank_unit,
    prepare_bank_answer_workspace,
)

_logger = logging.getLogger("routers.bank")

# 記憶體批改結果（鍵為 job_id）；供 GET /bank/qa/answer-result/{job_id} 輪詢。
_bank_answer_job_results: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# 共用查詢輔助
# ---------------------------------------------------------------------------


def _fetch_bank_group_row(supabase, bank_group_id: int, course_id: int, *, cols: str = "*") -> dict | None:
    """依 bank_group_id 取未刪除 Bank_Group 一列；course_id 不存在時自動略過篩選。"""

    def build(with_course_filter: bool):
        c = select_without_course_id_if_needed("Bank_Group", cols, with_course_filter)
        q = (
            supabase.table("Bank_Group")
            .select(c)
            .eq("bank_group_id", bank_group_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    sel = execute_with_course_id_fallback("Bank_Group", build, course_id)
    return sel.data[0] if sel.data else None


def require_bank_group_owner(
    supabase,
    bank_group_id: int,
    course_id: int,
    caller_person_id: str,
    *,
    cols: str = "*",
    forbidden_detail: str = "無權存取該 Bank_Group",
) -> dict:
    """取 Bank_Group 並驗證擁有者；不存在回 404、非擁有者回 403（detail 依呼叫情境傳入）。"""
    group = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols=cols)
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 bank_group_id={bank_group_id} 的 Bank_Group，或已刪除")
    if (group.get("person_id") or "").strip() != caller_person_id:
        raise HTTPException(status_code=403, detail=forbidden_detail)
    return group


def _bank_qa_rows_for_group(supabase, bank_group_id: int, course_id: int, *, cols: str = "*") -> list[dict]:
    """依 bank_group_id 取所有未刪除 Bank_QA，依 question_series_index、created_at 升序。"""

    def build(with_course_filter: bool):
        c = select_without_course_id_if_needed("Bank_QA", cols, with_course_filter)
        q = (
            supabase.table("Bank_QA")
            .select(c)
            .eq("bank_group_id", bank_group_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("question_series_index", desc=False).order("created_at", desc=False)

    sel = execute_with_course_id_fallback("Bank_QA", build, course_id)
    return sel.data or []


def _fetch_bank_qa_row(supabase, bank_qa_id: int, course_id: int, *, cols: str = "*") -> dict | None:
    """依 bank_qa_id 取未刪除 Bank_QA 一列。"""

    def build(with_course_filter: bool):
        c = select_without_course_id_if_needed("Bank_QA", cols, with_course_filter)
        q = (
            supabase.table("Bank_QA")
            .select(c)
            .eq("bank_qa_id", bank_qa_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    sel = execute_with_course_id_fallback("Bank_QA", build, course_id)
    return sel.data[0] if sel.data else None


# 單元內容存取（fetch_bank_unit_for_llm／rag_zip_page_id_from_unit）已抽至 .question_content，bank 與 quiz 共用。


# ---------------------------------------------------------------------------
# 題組單元解析（建立／列表題組時用，巢狀於 page/unit 之下）
# ---------------------------------------------------------------------------


def fetch_bank_unit_in_page(supabase, *, bank_page_id: str, bank_unit_id: int, course_id: int) -> dict:
    """取 Bank_Unit 並驗證其隸屬於路徑 bank_page_id（題組建立／列表巢狀於 page/unit 下時用）。"""
    unit = fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"找不到 bank_unit_id={bank_unit_id} 的 Bank_Unit，或已刪除")
    up = (unit.get("bank_page_id") or "").strip()
    if up and up != (bank_page_id or "").strip():
        raise HTTPException(status_code=404, detail="bank_unit_id 不屬於該 bank_page_id")
    return unit


# ---------------------------------------------------------------------------
# 逐題出題（LLM；同步，與 rag llm-generate 一致）
# ---------------------------------------------------------------------------


def _resolve_bank_quiz_llm_params(group: dict, course_id: int) -> tuple[str, str, str, str]:
    """解析出題所需 api_key／llm_model／qup／qsp（不呼叫 LLM）；prompt 一律取自 Bank_Group。"""
    api_key = get_bank_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 Bank API Key：PUT /v1/bank/llm-api-key（Course_Setting key=bank-api-key，依 course_id）",
        )
    llm_model = (group.get("question_llm_model") or "").strip() or get_bank_llm_model(course_id)
    qup = (group.get("question_user_prompt_text") or "").strip()
    qsp = (group.get("question_system_prompt_text") or "").strip()
    return api_key, llm_model, qup, qsp


# _generate_bank_quiz_fields 已抽至 .question_content.generate_question_fields_from_bank_unit（bank 與 quiz 共用）。


def bank_llm_generate_qa_impl(
    *,
    bank_group_id: int,
    caller_person_id: str,
    course_id: int,
):
    """在題組內產生下一題：依題組 prompt 與既有題目（勿重複）出一題，寫入 Bank_QA。"""
    supabase = get_supabase()
    group = require_bank_group_owner(
        supabase, bank_group_id, course_id, caller_person_id, forbidden_detail="無權於該題組出題"
    )
    pid = (group.get("person_id") or "").strip()

    qa_count = normalize_qa_count(group.get("qa_count"))

    existing = _bank_qa_rows_for_group(
        supabase, bank_group_id, course_id, cols="bank_qa_id, question_series_index, question_content, course_id"
    )
    if len(existing) >= qa_count:
        raise HTTPException(
            status_code=409,
            detail=f"本題組已達 qa_count 上限（{qa_count} 題），無法再出題",
        )

    api_key, llm_model, qup, qsp = _resolve_bank_quiz_llm_params(group, course_id)

    # 既有題幹 → 已出過題目（連續出題、勿重複）
    prior_items = [
        {"quiz_content": (q.get("question_content") or "").strip()}
        for q in existing
        if (q.get("question_content") or "").strip()
    ]

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
        )
        series_index = len(existing) + 1
        ts = now_taipei_iso()
        # prompt 為出題當下自題組複製、凍結（question_*、answer_user_prompt_text）。
        # 模型欄位記「實際呼叫 LLM 用的模型」：question_llm_model 為本次出題所用；
        # answer_llm_model 不在此存，於批改完成時寫入批改實際用的模型。
        qa_row: dict[str, Any] = {
            "bank_page_id": fields["bank_page_id"],
            "bank_unit_id": fields["bank_unit_id"],
            "bank_group_id": bank_group_id,
            "person_id": pid,
            "course_id": course_id,
            "question_series_index": series_index,
            "question_system_prompt_text": qsp,
            "question_user_prompt_text": qup,
            "question_content": fields["question_content"],
            "question_hint": fields["question_hint"],
            "question_answer_reference": fields["question_answer_reference"],
            "question_reason": fields["question_reason"],
            "question_llm_model": llm_model,
            "answer_user_prompt_text": (group.get("answer_user_prompt_text") or ""),
            "answer_content": "",
            "answer_critique": None,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = insert_bank_child_row("Bank_QA", qa_row)
        if not ins.data:
            raise HTTPException(status_code=500, detail="寫入 Bank_QA 失敗（無回傳資料）")
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        if is_llm_call_error(e):
            return llm_error_json_response(
                {
                    "llm_error": format_llm_error(e),
                    "bank_group_id": bank_group_id,
                    "question_content": "",
                    "question_hint": "",
                    "question_answer_reference": "",
                    "question_llm_model": llm_model,
                }
            )
        raise HTTPException(status_code=500, detail=str(e))


def bank_llm_regenerate_qa_impl(
    *,
    bank_qa_id: int,
    caller_person_id: str,
    course_id: int,
):
    """重新產生**同一題**（原地覆寫同一 bank_qa_id）。

    與 llm-generate 不同：不刪除、不新增任何 Bank_QA，亦不改動 question_series_index；只把這一題的
    question_* 內容用 LLM 重新產生並 UPDATE 回原列。「勿重複」清單為題組內**序號在本題之前**的題
    （question_series_index 較小者；與逐題出題一致，第 N 題只看 1..N-1）。重出後本題舊的作答／批改失效，一併清空。不檢查 qa_count 上限
    （重出佔用既有名額，非新增）。
    """
    supabase = get_supabase()
    qa = _fetch_bank_qa_row(supabase, bank_qa_id, course_id, cols="*")
    if not qa:
        raise HTTPException(status_code=404, detail=f"找不到 bank_qa_id={bank_qa_id} 的 Bank_QA，或已刪除")
    bank_group_id = int(qa.get("bank_group_id") or 0)
    if bank_group_id <= 0:
        raise HTTPException(status_code=400, detail="該題的 bank_group_id 無效")
    group = require_bank_group_owner(
        supabase, bank_group_id, course_id, caller_person_id, forbidden_detail="無權重出該題"
    )

    existing = _bank_qa_rows_for_group(
        supabase, bank_group_id, course_id, cols="bank_qa_id, question_series_index, question_content, course_id"
    )
    # 勿重複：只取「此題之前」的題（question_series_index 較小者）。
    # 與逐題出題一致——第 N 題只看 1..N-1；故重出第 1 題時無歷史 qa，即使後面已出多題也不納入。
    try:
        current_series_index = int(qa.get("question_series_index") or 0)
    except (TypeError, ValueError):
        current_series_index = 0
    prior_items = [
        {"quiz_content": (q.get("question_content") or "").strip()}
        for q in existing
        if int(q.get("bank_qa_id") or 0) != bank_qa_id
        and (current_series_index <= 0 or int(q.get("question_series_index") or 0) < current_series_index)
        and (q.get("question_content") or "").strip()
    ]
    api_key, llm_model, qup, qsp = _resolve_bank_quiz_llm_params(group, course_id)

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
            # 重出後舊作答／批改失效，一併清空（question_series_index 不變）；
            # 批改規則紀錄重置為題組現值（同新出一題），待下次批改完成再寫入實際使用值
            "answer_content": "",
            "answer_critique": None,
            "answer_llm_model": None,
            "answer_user_prompt_text": (group.get("answer_user_prompt_text") or ""),
            "updated_at": ts,
        }
        supabase.table("Bank_QA").update(update_row).eq("bank_qa_id", bank_qa_id).eq(
            "deleted", False
        ).execute()
        updated = _fetch_bank_qa_row(supabase, bank_qa_id, course_id, cols="*") or {**qa, **update_row}
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        if is_llm_call_error(e):
            return llm_error_json_response(
                {
                    "llm_error": format_llm_error(e),
                    "bank_group_id": bank_group_id,
                    "bank_qa_id": bank_qa_id,
                    "question_content": "",
                    "question_hint": "",
                    "question_answer_reference": "",
                    "question_llm_model": llm_model,
                }
            )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 批改（LLM；非同步，與 rag llm-answer 一致）
# ---------------------------------------------------------------------------


def update_bank_qa_with_answer(
    result_dict: dict,
    answer_content: str,
    *,
    bank_qa_id: int,
    answer_llm_model: str = "",
    answer_user_prompt_text: str = "",
) -> tuple[str, int] | None:
    """背景批改完成後更新 Bank_QA：寫 answer_content、answer_critique，並把本次批改**實際使用**的
    answer_llm_model／answer_user_prompt_text 寫回（QA 列記「這一題各次呼叫實際用了什麼」：
    出題規則於出題時寫入、批改規則於批改時寫入）；question_* 為出題時凍結，不覆寫。回傳 (id_key, id_val)。"""
    if bank_qa_id <= 0:
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
        supabase.table("Bank_QA").update(row).eq("bank_qa_id", bank_qa_id).eq("deleted", False).execute()
        chk = (
            supabase.table("Bank_QA")
            .select("answer_critique")
            .eq("bank_qa_id", bank_qa_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not chk.data:
            _logger.warning("Bank_QA update 後讀不到列（bank_qa_id=%s 不存在、已刪除或遭 RLS 擋）", bank_qa_id)
            return None
        if chk.data[0].get("answer_critique") is None:
            _logger.warning("Bank_QA 讀回 answer_critique 為空（bank_qa_id=%s）", bank_qa_id)
            return None
        return ("bank_qa_id", bank_qa_id)
    except Exception as e:
        _logger.warning("Bank_QA answer update 失敗: %s", e, exc_info=True)
    return None


async def enqueue_bank_qa_answer_job(
    background_tasks: BackgroundTasks,
    caller_person_id: str,
    course_id: int,
    *,
    bank_qa_id: int,
    answer_content: str,
) -> JSONResponse:
    """將 Bank_QA 批改排入 BackgroundTasks；回傳 202 + job_id（輪詢 GET /bank/qa/answer-result/{job_id}）。"""
    if bank_qa_id <= 0:
        return JSONResponse(status_code=400, content={"error": "bank_qa_id 必填且須為大於 0 的整數"})

    supabase = get_supabase()
    qa = _fetch_bank_qa_row(
        supabase,
        bank_qa_id,
        course_id,
        cols=(
            "bank_qa_id, bank_group_id, bank_unit_id, person_id, question_content, "
            "question_user_prompt_text, answer_user_prompt_text, answer_llm_model, course_id"
        ),
    )
    if not qa:
        return JSONResponse(status_code=404, content={"error": f"找不到 bank_qa_id={bank_qa_id} 的 Bank_QA"})
    pid = (qa.get("person_id") or "").strip()
    if pid != caller_person_id:
        return JSONResponse(status_code=403, content={"error": "無權批改該 Bank_QA"})

    quiz_content = (qa.get("question_content") or "").strip()
    if not quiz_content:
        return JSONResponse(
            status_code=400,
            content={"error": "該 Bank_QA 尚無題幹（question_content 為空），請先出題後再批改"},
        )

    # 批改設定一律用「題組現值」：規則／模型每次批改時自 Bank_Group 重抓（保證最新）；
    # 題組規則為空才回退此題出題當下凍結的快照（舊資料相容）。
    # 模型不看 QA 凍結值：題組 answer_llm_model 空 → 課程 bank-llm-model 現值；
    # 批改完成後由 update_bank_qa_with_answer 把實際使用的模型／批改規則寫回
    # Bank_QA.answer_llm_model／answer_user_prompt_text（QA 列記各次呼叫實際使用值）。
    bank_group_id = int(qa.get("bank_group_id") or 0)
    group = (
        _fetch_bank_group_row(
            supabase,
            bank_group_id,
            course_id,
            cols="bank_group_id, question_user_prompt_text, answer_user_prompt_text, answer_llm_model, course_id",
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

    api_key = get_bank_api_key(course_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "請設定 Bank API Key：PUT /v1/bank/llm-api-key（Course_Setting key=bank-api-key，依 course_id）"},
        )
    llm_model = (group.get("answer_llm_model") or "").strip() or get_bank_llm_model(course_id)

    bank_unit_id = int(qa.get("bank_unit_id") or 0)
    try:
        unit_type, transcript_answer, work_dir = prepare_bank_answer_workspace(
            supabase, bank_unit_id=bank_unit_id, course_id=course_id, prefix="myquizai_bank_answer"
        )
    except BankAnswerSetupError as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    job_id = str(uuid.uuid4())
    _bank_answer_job_results[job_id] = {"status": "pending", "result": None, "error": None, "llm_error": None}

    def insert_fn(rd, ans):
        return update_bank_qa_with_answer(
            rd,
            ans,
            bank_qa_id=bank_qa_id,
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
        _bank_answer_job_results,
        insert_fn,
        answer_user_prompt,
        bank_qa_id=bank_qa_id,
        unit_type=unit_type,
        transcript_answer=transcript_answer,
        quiz_user_prompt_text=quiz_user_prompt,
        llm_model=llm_model,
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "answer_llm_model": llm_model})


# ---------------------------------------------------------------------------
# GET 巢狀（題組與題目掛在單元下）
# ---------------------------------------------------------------------------


def _qas_by_bank_group_ids(group_ids: list[int], *, course_id: int | None = None) -> dict[int, list[dict]]:
    """依 bank_group_id 取未刪除 Bank_QA，回傳 bank_group_id -> qas[]（依 question_series_index 升序）。"""
    if not group_ids:
        return {}
    supabase = get_supabase()

    def build(with_course_filter: bool):
        q = (
            supabase.table("Bank_QA")
            .select("*")
            .in_("bank_group_id", group_ids)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("question_series_index", desc=False).order("created_at", desc=False)

    resp = execute_with_course_id_fallback("Bank_QA", build, course_id)
    out: dict[int, list[dict]] = {gid: [] for gid in group_ids}
    for row in resp.data or []:
        gid = row.get("bank_group_id")
        if gid is not None:
            try:
                out.setdefault(int(gid), []).append(row)
            except (TypeError, ValueError):
                pass
    return out


def groups_by_bank_unit_ids(unit_ids: list[int], *, course_id: int | None = None) -> dict[int, list[dict]]:
    """依 bank_unit_id 取未刪除 Bank_Group（含其 Bank_QA），回傳 bank_unit_id -> groups[]。"""
    if not unit_ids:
        return {}
    supabase = get_supabase()

    def build(with_course_filter: bool):
        q = (
            supabase.table("Bank_Group")
            .select("*")
            .in_("bank_unit_id", unit_ids)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.order("created_at", desc=False)

    resp = execute_with_course_id_fallback("Bank_Group", build, course_id)
    groups = resp.data or []

    group_ids: list[int] = []
    for g in groups:
        gid = g.get("bank_group_id")
        if gid is not None:
            try:
                group_ids.append(int(gid))
            except (TypeError, ValueError):
                pass
    group_ids = list(dict.fromkeys(group_ids))
    qas_by_group = _qas_by_bank_group_ids(group_ids, course_id=course_id)

    out: dict[int, list[dict]] = {uid: [] for uid in unit_ids}
    for g in groups:
        gid = g.get("bank_group_id")
        gid_int = int(gid) if gid is not None else None
        g["qas"] = qas_by_group.get(gid_int, []) if gid_int is not None else []
        uid = g.get("bank_unit_id")
        if uid is not None:
            try:
                out.setdefault(int(uid), []).append(g)
            except (TypeError, ValueError):
                pass
    return out
