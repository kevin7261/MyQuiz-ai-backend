"""routers.exam helpers（自 exam.py 拆分）。"""

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response
from postgrest.exceptions import APIError


from services.quiz_generation import (
    format_quiz_history_prompt_for_llm,
    generate_quiz,
    generate_quiz_followup,
    generate_quiz_followup_transcript_only,
    generate_quiz_transcript_only,
)

from services.exam_queries import (
    apply_exam_quiz_not_deleted,
    ensure_exam_quiz_rag_id_keys,
    enrich_exam_quizzes_rag_tab_from_units,
    select_rag_row_with_transcript_fallback,
)
from services.answering import (
    _rag_quiz_missing_column_error,
)
from utils.llm_error import format_llm_error, is_llm_call_error, llm_error_json_response
from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.serialization import to_json_safe
from utils.llm_key import get_exam_api_key, get_rag_llm_model
from utils.exam_course import require_exam_row
from utils.rag_course import (
    assert_row_course_id,
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.rag_exam_setting import rag_id_from_rag_page_id
from utils.rag_stem import get_rag_stem_from_rag_id, instruction_from_rag_row, transcript_from_row
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
from .schemas import ExamQuizHistoryPair, ExamQuizHistoryPromptFollowup, ExamQuizHistoryPromptStem

_logger = logging.getLogger("routers.exam")


# ---------------------------------------------------------------------------
# 路由內輔助（僅限此模組）
# ---------------------------------------------------------------------------


def _load_exam_for_quiz(
    supabase: Any,
    *,
    exam_id: int,
    exam_page_id: str,
    caller_person_id: str,
    course_id: int,
) -> tuple[str, str]:
    """回傳 (exam_page_id, person_id)。需擇一傳入 exam_id 或 exam_page_id；須符合 course_id。"""
    _ = supabase
    row = require_exam_row(
        course_id=course_id,
        exam_id=exam_id,
        exam_page_id=exam_page_id,
        person_id=caller_person_id,
    )
    out_tab = (row.get("exam_page_id") or "").strip()
    person_id = (row.get("person_id") or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="該 Exam 的 person_id 為空")
    return out_tab, person_id


def _exam_llm_generate_api_instruction(
    *,
    exam_quiz_id: int,
    exam_page_id: str | None,
    rag_page_id: str | None,
    rag_unit_id: int,
    rag_quiz_id: int | None,
    person_id: str | None,
    unit_name: str | None,
    quiz_name: str | None,
    quiz_user_prompt_text: str,
) -> str:
    """
    組出 POST /exam/quizzes/llm-generate 送進 utils.generate_quiz* 的 quiz_user_prompt_text 前綴。
    參數名與順序同 public.Exam_Quiz（至 quiz_user_prompt_text）；rag_quiz_id 列於提示時可為未關聯說明字串。
    """
    try:
        rqi = int(rag_quiz_id) if rag_quiz_id is not None else 0
    except (TypeError, ValueError):
        rqi = 0
    rag_quiz_id_line = f"`{rqi}`" if rqi > 0 else "（Exam_Quiz 未關聯 rag_quiz_id）"
    exam_page_id_s = (exam_page_id or "").strip() or "（未提供）"
    rag_page_id_s = (rag_page_id or "").strip() or "（未提供）"
    person_id_s = (person_id or "").strip() or "（未提供）"
    unit_name_s = (unit_name or "").strip() or "（未提供）"
    quiz_name_s = (quiz_name or "").strip() or "（未提供）"
    quiz_user_prompt_text_s = (quiz_user_prompt_text or "").strip() or "（未提供）"
    ru = int(rag_unit_id or 0)
    return textwrap.dedent(f"""
        ## 本次請求 API 參數

        請一併納入出題考量（欄位順序同 public.Exam_Quiz：exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, unit_name, quiz_name, quiz_user_prompt_text, …）。

        - **exam_quiz_id**：`{exam_quiz_id}`
        - **exam_page_id**：{exam_page_id_s}
        - **rag_page_id**：{rag_page_id_s}
        - **rag_unit_id**：`{ru}`
        - **rag_quiz_id**：{rag_quiz_id_line}
        - **person_id**：{person_id_s}
        - **unit_name**：{unit_name_s}
        - **quiz_name**：{quiz_name_s}

        ### quiz_user_prompt_text

        {quiz_user_prompt_text_s}
        """).strip()


def _create_exam_quiz_record(
    *,
    exam_page_id: str,
    caller_person_id: str,
    course_id: int,
) -> dict[str, Any]:
    """新增空白 Exam_Quiz；回傳 enrich 後列。"""
    supabase = get_supabase()
    resolved_page_id, person_id = _load_exam_for_quiz(
        supabase,
        exam_id=0,
        exam_page_id=exam_page_id,
        caller_person_id=caller_person_id,
        course_id=course_id,
    )
    qts = now_taipei_iso()
    quiz_row: dict[str, Any] = {
        "exam_page_id": resolved_page_id,
        "rag_page_id": "",
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
        "answer_rate": 0,
        "deleted": False,
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


def _prewrite_exam_quiz_history_fields(
    supabase: Any,
    *,
    exam_quiz_id: int,
    qa_dicts: list[dict[str, Any]],
    quiz_history_list_prompt_text: str,
) -> None:
    """出題前寫入 Exam_Quiz.quiz_history_list 與 quiz_history_list_prompt_text（皆為 JSON 字串）。"""
    payload: dict[str, Any] = {
        "quiz_history_list": serialize_rag_quiz_history_list(qa_dicts),
        "quiz_history_list_prompt_text": quiz_history_list_prompt_text or "[]",
        "updated_at": now_taipei_iso(),
    }
    try:
        for _ in range(4):
            try:
                supabase.table("Exam_Quiz").update(payload).eq("exam_quiz_id", exam_quiz_id).execute()
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
            "Exam_Quiz 預寫 quiz_history 欄位略過 exam_quiz_id=%s: %s", exam_quiz_id, e
        )


def _exam_quiz_history_qa_dicts(pairs: list[ExamQuizHistoryPair]) -> list[dict[str, Any]]:
    return parse_rag_quiz_history_list([p.model_dump() for p in pairs])


def _exam_quiz_history_prompt_dicts(
    pairs: list[ExamQuizHistoryPromptStem] | list[ExamQuizHistoryPromptFollowup],
    *,
    followup: bool,
) -> list[dict[str, Any]]:
    return parse_quiz_history_prompt_text([p.model_dump() for p in pairs], followup=followup)


def _resolve_exam_followup_mode(
    *,
    followup_requested: bool,
    follow_up_exam_quiz_id: int,
    exam_quiz_id: int,
    quiz_history: list[ExamQuizHistoryPair] | None,
    prompt_dicts: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool, int, list[dict[str, Any]]]:
    """
    follow_up_exam_quiz_id 以請求傳入為準。
    回傳 (use_followup_llm, mark_follow_up, follow_up_exam_quiz_id, qa_dicts)。

    mark_follow_up：followup 端點且 follow_up_exam_quiz_id>0 → 寫入 follow_up=true。
    use_followup_llm：mark_follow_up 且 quiz_history_list_prompt_text 非空 → 使用追問 LLM prompt。
    """
    request_qa = _exam_quiz_history_qa_dicts(quiz_history or [])

    if not followup_requested:
        return False, False, 0, request_qa

    resolved_id = int(follow_up_exam_quiz_id or 0)
    if resolved_id <= 0 or resolved_id == exam_quiz_id:
        return False, False, 0, request_qa

    mark_follow_up = True
    use_followup_llm = bool(prompt_dicts)
    return use_followup_llm, mark_follow_up, resolved_id, request_qa


def _select_rag_unit_for_exam_prompt(
    supabase: Any,
    *,
    rag_page_id_for_units: str,
    course_id: int,
    stem_rag_unit_id: int | None,
    unit_filter: str | None,
) -> dict | None:
    """依 rag_page_id 列出 Rag_Unit（含欄位降級相容），挑出對應 stem_rag_unit_id 或 unit_filter 的單元；無 rag_page_id 時回傳 None。"""
    selected: dict | None = None
    if rag_page_id_for_units:
        def _unit_q_select(cols: str, with_course_filter: bool):
            c = select_without_course_id_if_needed("Rag_Unit", cols, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(c)
                .eq("rag_page_id", rag_page_id_for_units)
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
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _cols_no_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _cols_no_fc = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcript"
        )
        _cols_min = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap"
        )
        _cols_legacy_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, folder_combination, unit_type, repack_file_name, rag_file_name, "
            "rag_file_size, rag_chunk_size, rag_chunk_overlap, transcription"
        )
        _cols_no_fc_legacy_tr = (
            "rag_unit_id, rag_page_id, person_id, unit_name, unit_type, repack_file_name, rag_file_name, "
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
    rag_page_id: str,
    rag_unit_id: int,
    rag_quiz_id: int,
    caller_person_id: str,
    course_id: int,
    followup: bool,
    quiz_history: list[ExamQuizHistoryPair] | None = None,
    quiz_history_list_prompt_items: list[dict[str, Any]] | None = None,
    follow_up_exam_quiz_id: int = 0,
    always_mark_follow_up: bool = False,
    quiz_system_prompt_text: str = "",
):
    supabase = get_supabase()
    try:
        qsel = apply_exam_quiz_not_deleted(
            supabase.table("Exam_Quiz")
            .select(
                "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
                "unit_name, quiz_name, quiz_history_list, created_at"
            )
            .eq("exam_quiz_id", exam_quiz_id)
            .eq("course_id", course_id)
        ).limit(1).execute()
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "quiz_history_list" in msg:
            qsel = apply_exam_quiz_not_deleted(
                supabase.table("Exam_Quiz")
                .select(
                    "exam_quiz_id, exam_page_id, rag_page_id, rag_unit_id, rag_quiz_id, person_id, course_id, "
                    "unit_name, quiz_name, created_at"
                )
                .eq("exam_quiz_id", exam_quiz_id)
                .eq("course_id", course_id)
            ).limit(1).execute()
        else:
            raise
    if not qsel.data or len(qsel.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 exam_quiz_id={exam_quiz_id} 的 Exam_Quiz，或已刪除")
    qrow = qsel.data[0]
    person_id = (qrow.get("person_id") or "").strip()
    if person_id != caller_person_id:
        raise HTTPException(status_code=403, detail="無權對該 Exam_Quiz 出題")

    request_history = _exam_quiz_history_qa_dicts(quiz_history or [])
    qa_dicts, _ = resolve_quiz_history_for_generate(
        request_history=request_history,
    )
    prompt_dicts = list(quiz_history_list_prompt_items or [])
    prompt_db_str = serialize_quiz_history_prompt_text(prompt_dicts, followup=followup)
    prompt_for_llm = format_quiz_history_prompt_for_llm(prompt_dicts, followup=followup)
    _prewrite_exam_quiz_history_fields(
        supabase,
        exam_quiz_id=exam_quiz_id,
        qa_dicts=qa_dicts,
        quiz_history_list_prompt_text=prompt_db_str,
    )

    use_followup_llm, mark_follow_up, resolved_follow_up_id, _ = (
        _resolve_exam_followup_mode(
            followup_requested=followup,
            follow_up_exam_quiz_id=follow_up_exam_quiz_id,
            exam_quiz_id=exam_quiz_id,
            quiz_history=quiz_history,
            prompt_dicts=prompt_dicts,
        )
    )
    if always_mark_follow_up:
        mark_follow_up = True
        resolved_follow_up_id = int(follow_up_exam_quiz_id or 0)
        if resolved_follow_up_id == exam_quiz_id:
            resolved_follow_up_id = 0

    if followup and mark_follow_up:
        use_followup_llm = bool(prompt_dicts)
        if use_followup_llm:
            prompt_for_llm = format_quiz_history_prompt_for_llm(prompt_dicts, followup=True)

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

    tab_strip = (rag_page_id or "").strip()
    if not tab_strip:
        raise HTTPException(status_code=400, detail="rag_page_id 不可為空白")

    row_rtab = ""
    _row_rt = qrow.get("rag_page_id")
    if _row_rt is not None:
        row_rtab = str(_row_rt).strip()
    if row_rtab and row_rtab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_page_id 須與該 Exam_Quiz 列已存 rag_page_id 一致；"
                f"請求為 {tab_strip!r}，列上為 {row_rtab!r}"
            ),
        )

    def fetch_ru_one(*, include_folder_combination: bool):
        def build(with_course_filter: bool):
            base_cols = (
                "rag_page_id, unit_name, folder_combination, course_id"
                if include_folder_combination
                else "rag_page_id, unit_name, course_id"
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
    ru_tab = (ru_one.data[0].get("rag_page_id") or "").strip()
    if ru_tab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_page_id 須與 rag_unit_id 所隸 Rag Tab 一致；"
                f"請求為 {tab_strip!r}，Rag_Unit 為 {ru_tab!r}"
            ),
        )

    def build_rq_one(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Rag_Quiz",
            "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id, quiz_name, quiz_user_prompt_text, "
            "quiz_system_prompt_text, quiz_content, quiz_hint, quiz_answer_reference, answer_user_prompt_text",
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
    rq_tab = (rq_row0.get("rag_page_id") or "").strip()
    if rq_tab != tab_strip:
        raise HTTPException(
            status_code=400,
            detail=(
                "請求 rag_page_id 須與 rag_quiz_id 所隸 Rag Tab 一致；"
                f"請求為 {tab_strip!r}，Rag_Quiz 為 {rq_tab!r}"
            ),
        )

    rag_id_resolved = rag_id_from_rag_page_id(supabase, tab_strip, course_id)
    if rag_id_resolved is None or rag_id_resolved <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 rag_page_id={tab_strip!r} 對應之 Rag（deleted=false）",
        )

    quiz_user_prompt_resolved = (rq_row0.get("quiz_user_prompt_text") or "").strip()
    answer_user_prompt_resolved = (rq_row0.get("answer_user_prompt_text") or "").strip()
    # 僅 followup（接續出題）採用 quiz_system_prompt_text；一般出題不套用、不寫入（連來源 Rag_Quiz 既存值也不繼承）。
    quiz_system_prompt_resolved = (
        (
            (quiz_system_prompt_text or "").strip()
            or (rq_row0.get("quiz_system_prompt_text") or "").strip()
        )
        if followup
        else ""
    )

    _ru0 = ru_one.data[0]
    _ru_display = (_ru0.get("unit_name") or "").strip()
    _ru_folder = (_ru0.get("folder_combination") or "").strip()
    unit_filter: str | None = (_ru_folder or _ru_display or "").strip() or None
    stem_rag_unit_id: int | None = effective_ruid if effective_ruid > 0 else None
    if not unit_filter:
        unit_filter = (qrow.get("unit_name") or "").strip() or None

    api_key = get_exam_api_key(course_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="請設定 Exam API Key：PUT /exam/llm-api-key（Course_Setting key=exam-api-key，依 course_id）",
        )
    llm_model = get_rag_llm_model(course_id)

    rag_rows = select_rag_row_with_transcript_fallback(supabase, rag_id_resolved)
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id_resolved} 的 Rag 資料，或已刪除")
    rag_row = rag_rows.data[0]
    assert_row_course_id(rag_row, course_id, "Rag")
    rag_id = int(rag_row.get("rag_id") or 0)
    rag_page_id_for_units = (rag_row.get("rag_page_id") or "").strip()

    stem, rag_zip_page_id = get_rag_stem_from_rag_id(
        supabase, rag_id, unit_name=unit_filter, rag_unit_id=stem_rag_unit_id
    )

    selected = _select_rag_unit_for_exam_prompt(
        supabase,
        rag_page_id_for_units=rag_page_id_for_units,
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
            detail="單元類型 2／3／4 需有逐字稿：請於 Rag_Unit 設定 transcript，或經 POST /v1/rag/pages/{rag_page_id}/build-zip 寫入",
        )

    prompt_rag_unit_id = int(selected.get("rag_unit_id") or 0) if selected else int(qrow.get("rag_unit_id") or 0)
    prompt_rag_qid = cand_rag_qid
    un_for_prompt = (qrow.get("unit_name") or "").strip() or None
    rq_quiz_name = (rq_row0.get("quiz_name") or "").strip()
    qn_for_prompt = (qrow.get("quiz_name") or "").strip() or rq_quiz_name or None
    api_instr = _exam_llm_generate_api_instruction(
        exam_quiz_id=exam_quiz_id,
        exam_page_id=(qrow.get("exam_page_id") or "").strip() or None,
        rag_page_id=tab_strip,
        rag_unit_id=prompt_rag_unit_id,
        rag_quiz_id=prompt_rag_qid,
        person_id=person_id,
        unit_name=un_for_prompt,
        quiz_name=qn_for_prompt,
        quiz_user_prompt_text=quiz_user_prompt_resolved,
    )

    path: Path | None = None
    try:
        if unit_type_val in (2, 3, 4):
            if use_followup_llm:
                result = generate_quiz_followup_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                    quiz_system_prompt_text=quiz_system_prompt_resolved,
                )
            else:
                result = generate_quiz_transcript_only(
                    api_key=api_key,
                    transcript=transcript_text,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                    quiz_system_prompt_text=quiz_system_prompt_resolved,
                )
        else:
            path = get_zip_path(rag_zip_page_id)
            if not path or not path.exists():
                raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 rag_id={rag_id}（page_id={rag_zip_page_id}）")
            if use_followup_llm:
                result = generate_quiz_followup(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                    quiz_system_prompt_text=quiz_system_prompt_resolved,
                )
            else:
                result = generate_quiz(
                    path,
                    api_key=api_key,
                    quiz_user_prompt_text=api_instr,
                    quiz_history_list_prompt_text=prompt_for_llm,
                    llm_model=llm_model,
                    quiz_system_prompt_text=quiz_system_prompt_resolved,
                )
        result["transcript"] = "" if unit_type_val == 1 else transcript_text
        result["rag_output"] = {"rag_page_id": stem, "unit_name": stem, "filename": f"{stem}.zip"}

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
        result["quiz_system_prompt_text"] = quiz_system_prompt_resolved
        result["answer_user_prompt_text"] = answer_user_prompt_resolved
        result["unit_name"] = unit_name_for_display
        quiz_update: dict[str, Any] = {
            "rag_page_id": tab_strip,
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
            "quiz_history_list": serialize_rag_quiz_history_list(qa_dicts),
            "quiz_history_list_prompt_text": prompt_db_str,
            "quiz_llm_model": llm_model,
            "updated_at": qts,
        }
        # 僅 followup 出題寫入 quiz_system_prompt_text；一般出題不更動該欄。
        if followup:
            quiz_update["quiz_system_prompt_text"] = quiz_system_prompt_resolved
        if mark_follow_up:
            quiz_update["follow_up"] = True
            quiz_update["follow_up_exam_quiz_id"] = resolved_follow_up_id
            result["follow_up"] = True
            result["follow_up_exam_quiz_id"] = resolved_follow_up_id
        else:
            quiz_update["follow_up"] = False
            quiz_update["follow_up_exam_quiz_id"] = 0
        if qa_dicts:
            result["quiz_history_list"] = qa_dicts
        result["quiz_history_list_prompt_text"] = prompt_dicts
        result["created_at"] = to_taipei_iso(qrow.get("created_at"))
        result["rag_page_id"] = tab_strip
        result["rag_unit_id"] = int(rag_unit_id)
        result["rag_quiz_id"] = int(rag_quiz_id)
        result["quiz_llm_model"] = llm_model
        log_path = (
            "/exam/quizzes/llm-generate-followup"
            if use_followup_llm
            else "/exam/quizzes/llm-generate"
        )
        try:
            update_payload = dict(quiz_update)
            for _ in range(4):
                try:
                    supabase.table("Exam_Quiz").update(update_payload).eq("exam_quiz_id", exam_quiz_id).execute()
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
                    if _rag_quiz_missing_column_error(upd_err, "quiz_system_prompt_text") and "quiz_system_prompt_text" in update_payload:
                        update_payload.pop("quiz_system_prompt_text")
                        continue
                    raise
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
        if is_llm_call_error(e):
            err_payload: dict[str, Any] = {
                "llm_error": format_llm_error(e),
                "exam_quiz_id": exam_quiz_id,
                "quiz_content": "",
                "quiz_hint": "",
                "quiz_answer_reference": "",
                "rag_page_id": tab_strip,
                "rag_unit_id": int(rag_unit_id),
                "rag_quiz_id": int(rag_quiz_id),
                "quiz_llm_model": llm_model,
            }
            if mark_follow_up:
                err_payload["follow_up"] = True
                err_payload["follow_up_exam_quiz_id"] = resolved_follow_up_id
            return llm_error_json_response(err_payload)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path is not None:
            safe_unlink(path)

