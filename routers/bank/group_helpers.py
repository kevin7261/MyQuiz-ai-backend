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
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response
from openai import OpenAI

from services.bank_generation import (
    BANK_QUIZ_LLM_MODEL,
    format_bank_quiz_history_prompt_for_llm,
    generate_bank_quiz,
    generate_bank_quiz_transcript_only,
)
from services.bank_answering import (
    answer_critique_plain_text_from_result,
    cleanup_answer_workspace,
    run_bank_answer_job_background,
)
from utils.bank_llm_error import format_llm_error, is_llm_call_error, llm_error_json_response
from utils.bank_llm_key import get_bank_api_key, get_bank_llm_model
from utils.taipei_time import now_taipei_iso
from utils.bank_stem import transcript_from_row
from utils.bank_course import (
    execute_with_course_id_fallback,
    insert_bank_child_row,
    select_without_course_id_if_needed,
)
from utils.supabase import get_supabase
from utils.bank_storage import get_zip_path
from utils.serialization import to_json_safe
from utils.fs import safe_unlink

from .helpers import (
    BANK_UNIT_TYPE_MP3,
    BANK_UNIT_TYPE_TEXT,
    BANK_UNIT_TYPE_YOUTUBE,
)

_logger = logging.getLogger("routers.bank")

# 記憶體批改結果（鍵為 job_id）；供 GET /bank/qa/answer-result/{job_id} 輪詢。
_bank_answer_job_results: dict[str, dict[str, Any]] = {}

_TRANSCRIPT_UNIT_TYPES = (BANK_UNIT_TYPE_TEXT, BANK_UNIT_TYPE_MP3, BANK_UNIT_TYPE_YOUTUBE)


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


def _fetch_bank_unit_for_llm(supabase, bank_unit_id: int, course_id: int) -> dict | None:
    """取出題／批改所需之 Bank_Unit 欄位（不含 folder_combination，避免舊表 42703）。"""

    def build(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Bank_Unit",
            "bank_unit_id, bank_page_id, person_id, unit_name, unit_type, transcript, upload_file_name, course_id",
            with_course_filter,
        )
        q = (
            supabase.table("Bank_Unit")
            .select(cols)
            .eq("bank_unit_id", bank_unit_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    sel = execute_with_course_id_fallback("Bank_Unit", build, course_id)
    return sel.data[0] if sel.data else None


def _rag_zip_page_id_from_unit(unit_row: dict) -> str:
    """由 Bank_Unit.upload_file_name（{stem}_rag.zip）取出 rag ZIP 的 page_id（{stem}_rag）。"""
    rf = (unit_row.get("upload_file_name") or "").strip()
    if rf.lower().endswith(".zip"):
        rid = rf[:-4].strip()
        if rid and "/" not in rid and "\\" not in rid:
            return rid
    return ""


# ---------------------------------------------------------------------------
# 題組單元解析（建立／列表題組時用，巢狀於 page/unit 之下）
# ---------------------------------------------------------------------------


def fetch_bank_unit_in_page(supabase, *, bank_page_id: str, bank_unit_id: int, course_id: int) -> dict:
    """取 Bank_Unit 並驗證其隸屬於路徑 bank_page_id（題組建立／列表巢狀於 page/unit 下時用）。"""
    unit = _fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"找不到 bank_unit_id={bank_unit_id} 的 Bank_Unit，或已刪除")
    up = (unit.get("bank_page_id") or "").strip()
    if up and up != (bank_page_id or "").strip():
        raise HTTPException(status_code=404, detail="bank_unit_id 不屬於該 bank_page_id")
    return unit


# ---------------------------------------------------------------------------
# 逐題出題（LLM；同步，與 rag llm-generate 一致）
# ---------------------------------------------------------------------------


def _generate_question_reason(
    api_key: str, *, question_content: str, quiz_answer_reference: str, llm_model: str | None = None
) -> str:
    """出題後產生「出題理由」（1–2 句，寫入 Bank_QA.question_reason）。失敗時回空字串，不影響出題。"""
    qc = (question_content or "").strip()
    if not qc or not (api_key or "").strip():
        return ""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=llm_model or BANK_QUIZ_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是出題教師。請用 1–2 句說明這道題目的出題理由（想考察的重點概念或能力），語種與題目一致。回傳一個 JSON 物件，鍵名固定為 question_reason（Markdown 字串）。",
                },
                {
                    "role": "user",
                    "content": f"## 題目\n\n{qc}\n\n## 參考答案\n\n{(quiz_answer_reference or '').strip() or '（未提供）'}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        if isinstance(data, dict):
            return (data.get("question_reason") or "").strip()
    except Exception:
        _logger.warning("產生 question_reason 失敗，留空", exc_info=True)
    return ""


def bank_llm_generate_qa_impl(
    *,
    bank_group_id: int,
    caller_person_id: str,
    course_id: int,
    question_user_prompt_override: str = "",
    question_system_prompt_override: str = "",
):
    """在題組內產生下一題：依題組 prompt 與既有題目（勿重複）出一題，寫入 Bank_QA。"""
    supabase = get_supabase()
    group = _fetch_bank_group_row(supabase, bank_group_id, course_id, cols="*")
    if not group:
        raise HTTPException(status_code=404, detail=f"找不到 bank_group_id={bank_group_id} 的 Bank_Group，或已刪除")
    pid = (group.get("person_id") or "").strip()
    if pid != caller_person_id:
        raise HTTPException(status_code=403, detail="無權於該題組出題")

    try:
        qa_count = int(group.get("qa_count") or 0)
    except (TypeError, ValueError):
        qa_count = 0

    existing = _bank_qa_rows_for_group(
        supabase, bank_group_id, course_id, cols="bank_qa_id, question_series_index, question_content, course_id"
    )
    if qa_count > 0 and len(existing) >= qa_count:
        raise HTTPException(
            status_code=409,
            detail=f"本題組已達 qa_count 上限（{qa_count} 題），無法再出題",
        )

    bank_unit_id = int(group.get("bank_unit_id") or 0)
    if bank_unit_id <= 0:
        raise HTTPException(status_code=400, detail="該題組對應的 bank_unit_id 無效")
    unit = _fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"找不到 bank_unit_id={bank_unit_id} 的 Bank_Unit")
    bank_page_id = (unit.get("bank_page_id") or group.get("bank_page_id") or "").strip()

    api_key = get_bank_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 Bank API Key：PUT /v1/bank/llm-api-key（Course_Setting key=bank-api-key，依 course_id）",
        )
    llm_model = (group.get("question_llm_model") or "").strip() or get_bank_llm_model(course_id)

    qup = (question_user_prompt_override or "").strip() or (group.get("question_user_prompt_text") or "").strip()
    qsp = (question_system_prompt_override or "").strip() or (group.get("question_system_prompt_text") or "").strip()

    # 既有題幹 → 已出過題目（連續出題、勿重複）
    prior_items = [
        {"quiz_content": (q.get("question_content") or "").strip()}
        for q in existing
        if (q.get("question_content") or "").strip()
    ]
    prompt_for_llm = format_bank_quiz_history_prompt_for_llm(prior_items)

    try:
        unit_type_val = int(unit.get("unit_type") or 0)
    except (TypeError, ValueError):
        unit_type_val = 0
    transcript_text = transcript_from_row(unit)

    path: Path | None = None
    try:
        if unit_type_val in _TRANSCRIPT_UNIT_TYPES:
            if not transcript_text:
                raise HTTPException(
                    status_code=400,
                    detail="單元類型 2／3／4 需有逐字稿：請於 Bank_Unit 設定 transcript，或經 build-zip 寫入",
                )
            result = generate_bank_quiz_transcript_only(
                api_key=api_key,
                transcript=transcript_text,
                quiz_user_prompt_text=qup,
                quiz_history_list_prompt_text=prompt_for_llm,
                llm_model=llm_model,
                quiz_system_prompt_text=qsp,
            )
        else:
            rag_zip_page_id = _rag_zip_page_id_from_unit(unit)
            if not rag_zip_page_id:
                raise HTTPException(
                    status_code=400,
                    detail="該單元尚無 RAG ZIP（upload_file_name 為空）；請先執行 build-zip 建立向量庫",
                )
            path = get_zip_path(rag_zip_page_id)
            if not path or not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP（page_id={rag_zip_page_id}），請確認該單元已 build-zip",
                )
            result = generate_bank_quiz(
                path,
                api_key=api_key,
                quiz_user_prompt_text=qup,
                quiz_history_list_prompt_text=prompt_for_llm,
                llm_model=llm_model,
                quiz_system_prompt_text=qsp,
            )

        qc = (result.get("quiz_content") or "").strip()
        qh = (result.get("quiz_hint") or "").strip()
        qref = (result.get("quiz_answer_reference") or "").strip()
        question_reason = _generate_question_reason(
            api_key, question_content=qc, quiz_answer_reference=qref, llm_model=llm_model
        )
        series_index = len(existing) + 1
        ts = now_taipei_iso()
        # prompt 為出題當下自題組複製、凍結（question_*、answer_user_prompt_text）。
        # 模型欄位記「實際呼叫 LLM 用的模型」：question_llm_model 為本次出題所用；
        # answer_llm_model 不在此存，於批改完成時寫入批改實際用的模型。
        qa_row: dict[str, Any] = {
            "bank_page_id": bank_page_id,
            "bank_unit_id": bank_unit_id,
            "bank_group_id": bank_group_id,
            "person_id": pid,
            "course_id": course_id,
            "question_series_index": series_index,
            "question_system_prompt_text": qsp,
            "question_user_prompt_text": qup,
            "question_content": qc,
            "question_hint": qh,
            "question_answer_reference": qref,
            "question_reason": question_reason,
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
    finally:
        if path is not None:
            safe_unlink(path)


# ---------------------------------------------------------------------------
# 批改（LLM；非同步，與 rag llm-answer 一致）
# ---------------------------------------------------------------------------


def update_bank_qa_with_answer(
    result_dict: dict,
    answer_content: str,
    *,
    bank_qa_id: int,
    answer_llm_model: str = "",
) -> tuple[str, int] | None:
    """背景批改完成後更新 Bank_QA：寫 answer_content、answer_critique 與 answer_llm_model（批改實際用的模型）；prompt 為出題時凍結，不覆寫。回傳 (id_key, id_val)。"""
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

    # 批改設定一律用「出題當下凍結在此題」的快照；舊資料（欄位空）才回退題組。
    bank_group_id = int(qa.get("bank_group_id") or 0)
    quiz_user_prompt = (qa.get("question_user_prompt_text") or "").strip()
    answer_user_prompt = (qa.get("answer_user_prompt_text") or "").strip()
    qa_answer_llm_model = (qa.get("answer_llm_model") or "").strip()
    if not (quiz_user_prompt and answer_user_prompt and qa_answer_llm_model):
        group = _fetch_bank_group_row(
            supabase,
            bank_group_id,
            course_id,
            cols="bank_group_id, question_user_prompt_text, answer_user_prompt_text, answer_llm_model, course_id",
        )
        if group:
            quiz_user_prompt = quiz_user_prompt or (group.get("question_user_prompt_text") or "").strip()
            answer_user_prompt = answer_user_prompt or (group.get("answer_user_prompt_text") or "").strip()
            qa_answer_llm_model = qa_answer_llm_model or (group.get("answer_llm_model") or "").strip()

    api_key = get_bank_api_key(course_id)
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "請設定 Bank API Key：PUT /v1/bank/llm-api-key（Course_Setting key=bank-api-key，依 course_id）"},
        )
    llm_model = qa_answer_llm_model or get_bank_llm_model(course_id)

    bank_unit_id = int(qa.get("bank_unit_id") or 0)
    unit = _fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id) if bank_unit_id > 0 else None
    try:
        unit_type = int(unit.get("unit_type") or 0) if unit else 0
    except (TypeError, ValueError):
        unit_type = 0
    transcript_text = transcript_from_row(unit) if unit else ""

    transcript_answer: str | None = None
    if unit_type in _TRANSCRIPT_UNIT_TYPES:
        if not transcript_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "批改用 transcript 未設定：請於 Bank_Unit 設定 transcript（單元 2／3／4）"},
            )
        transcript_answer = transcript_text
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_bank_answer_tx_"))
    else:
        rag_zip_page_id = _rag_zip_page_id_from_unit(unit or {})
        rag_zip_path = get_zip_path(rag_zip_page_id) if rag_zip_page_id else None
        if not rag_zip_path or not rag_zip_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"找不到 RAG ZIP（page_id={rag_zip_page_id}），請確認該單元已 build-zip"},
            )
        work_dir = Path(tempfile.mkdtemp(prefix="myquizai_bank_answer_"))
        zip_source_path = work_dir / "ref.zip"
        extract_folder = work_dir / "extract"
        extract_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(rag_zip_path, zip_source_path)
            if not zipfile.is_zipfile(zip_source_path):
                cleanup_answer_workspace(work_dir)
                return JSONResponse(status_code=400, content={"error": "無效的 ZIP 檔"})
        except Exception as e:
            cleanup_answer_workspace(work_dir)
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            safe_unlink(rag_zip_path)

    job_id = str(uuid.uuid4())
    _bank_answer_job_results[job_id] = {"status": "pending", "result": None, "error": None, "llm_error": None}

    def insert_fn(rd, ans):
        return update_bank_qa_with_answer(rd, ans, bank_qa_id=bank_qa_id, answer_llm_model=llm_model)

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
