"""routers.grade helpers（自 grade.py 拆分）。"""

import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from services.quiz_generation import (
    format_quiz_history_prompt_for_llm,
    generate_quiz,
    generate_quiz_followup,
    generate_quiz_followup_transcript_only,
    generate_quiz_transcript_only,
)

from fastapi import BackgroundTasks, HTTPException
from postgrest.exceptions import APIError
from fastapi.responses import JSONResponse, Response

from services.grading import (
    cleanup_grade_workspace,
    run_grade_job_background,
    update_rag_quiz_with_grade,
    _rag_quiz_missing_column_error,
)
from utils.llm_error import format_llm_error, is_llm_call_error, llm_error_json_response
from utils.llm_key import get_rag_api_key, get_rag_llm_model
from utils.taipei_time import now_taipei_iso
from utils.rag_stem import get_rag_stem_from_rag_id, instruction_from_rag_row, transcript_from_row
from utils.rag_course import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.supabase import get_supabase
from utils.zip_storage import get_zip_path
from utils.db_schema import (
    parse_quiz_history_prompt_text,
    parse_rag_quiz_history_list,
    resolve_quiz_history_for_generate,
    serialize_quiz_history_prompt_text,
    serialize_rag_quiz_history_list,
)


from utils.fs import safe_unlink
from .schemas import QuizHistoryPair, QuizHistoryPromptFollowup, QuizHistoryPromptStem

_logger = logging.getLogger("routers.grade")


_grade_job_results: dict[str, dict[str, Any]] = {}


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
            detail="請設定 RAG API Key：PUT /v1/rag/llm-api-key（Course_Setting key=rag-api-key，依 course_id）",
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
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 或 Rag 設定 transcript，或經 POST /v1/rag/pages/{rag_page_id}/build-zip 寫入 Rag_Unit.transcript",
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
        if is_llm_call_error(e):
            return llm_error_json_response(
                {
                    "llm_error": format_llm_error(e),
                    "rag_quiz_id": rag_quiz_id,
                    "quiz_content": "",
                    "quiz_hint": "",
                    "quiz_answer_reference": "",
                    "follow_up": followup,
                    "quiz_llm_model": llm_model,
                }
            )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path is not None:
            safe_unlink(path)


# ---------------------------------------------------------------------------
# POST /v1/rag/quizzes/llm-answer
# ---------------------------------------------------------------------------


async def _enqueue_rag_llm_answer_job(
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
    """將 RAG llm-answer 工作排入 BackgroundTasks；`answer-result` 輪詢鍵為記憶體 job_id。"""
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
                "error": "請設定 RAG API Key：PUT /v1/rag/llm-api-key（Course_Setting key=rag-api-key，依 course_id）",
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
            safe_unlink(rag_zip_path)

    job_id = str(uuid.uuid4())
    _grade_job_results[job_id] = {"status": "pending", "result": None, "error": None, "llm_error": None}
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
