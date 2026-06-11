"""
Quiz 模組資料查詢（供 user_analysis、quiz_analysis 使用）。

資料流：Quiz → Quiz_Group → Quiz_QA
- Quiz_QA.person_id：作答者（學生）
- Quiz_QA.quiz_page_id：試卷識別碼
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from utils.db_schema import (
    QUIZ_GROUP_SELECT_COLUMNS,
    QUIZ_GROUP_TABLE,
    QUIZ_QA_SELECT_COLUMNS,
    QUIZ_QA_TABLE,
    QUIZ_SELECT_COLUMNS,
    QUIZ_TABLE,
)
from utils.supabase import get_supabase

logger = logging.getLogger(__name__)


def normalize_quiz_qa(row: dict[str, Any]) -> dict[str, Any]:
    """
    將 Quiz_QA 列補充 weakness_report 共用函式所需欄位別名：
    - quiz_content ← question_content
    - quiz_answer_reference ← question_answer_reference
    - quiz_rate ← answer_rate
    確保 quiz_has_answer() 與 generate_weakness_report_md() 可直接使用此列。
    """
    return {
        **row,
        "quiz_content": row.get("question_content") or "",
        "quiz_answer_reference": row.get("question_answer_reference") or "",
        "quiz_rate": row.get("answer_rate"),
    }


def quiz_qas_by_person_id(
    person_id: str,
    course_id: int,
) -> list[dict[str, Any]]:
    """取得特定學生在某課程的所有 Quiz_QA（deleted=false，依 quiz_qa_id 升冪，正規化後）。"""
    pid = (person_id or "").strip()
    if not pid:
        return []
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_QA_TABLE)
            .select(QUIZ_QA_SELECT_COLUMNS)
            .eq("person_id", pid)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .order("quiz_qa_id", desc=False)
            .execute()
        )
        return [normalize_quiz_qa(row) for row in resp.data or []]
    except Exception:
        logger.exception(
            "quiz_qas_by_person_id failed person_id=%s course_id=%s", person_id, course_id
        )
        return []


def quiz_qas_by_course_id(course_id: int | str) -> list[dict[str, Any]]:
    """
    取得課程內所有學生的 Quiz_QA（deleted=false，依 person_id 再依 quiz_qa_id 升冪，正規化後）。
    用於 quiz_analysis（整門課程測驗作答分析）彙整全體學生答題狀況。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_QA_TABLE)
            .select(QUIZ_QA_SELECT_COLUMNS)
            .eq("course_id", int(course_id))
            .eq("deleted", False)
            .order("person_id", desc=False)
            .order("quiz_qa_id", desc=False)
            .execute()
        )
        return [normalize_quiz_qa(row) for row in resp.data or []]
    except Exception:
        logger.exception("quiz_qas_by_course_id failed course_id=%s", course_id)
        return []


def _fetch_tab_name_map(quiz_page_ids: list[str]) -> dict[str, str]:
    if not quiz_page_ids:
        return {}
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_TABLE)
            .select(QUIZ_SELECT_COLUMNS)
            .in_("quiz_page_id", quiz_page_ids)
            .execute()
        )
        return {r["quiz_page_id"]: r.get("tab_name") or "" for r in resp.data or []}
    except Exception:
        logger.exception("_fetch_tab_name_map failed quiz_page_ids=%s", quiz_page_ids)
        return {}


def _fetch_group_meta_map(group_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not group_ids:
        return {}
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(QUIZ_GROUP_TABLE)
            .select(QUIZ_GROUP_SELECT_COLUMNS)
            .in_("quiz_group_id", group_ids)
            .execute()
        )
        return {r["quiz_group_id"]: r for r in resp.data or []}
    except Exception:
        logger.exception("_fetch_group_meta_map failed group_ids=%s", group_ids)
        return {}


def quizzes_with_qas_response(
    qas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    將 Quiz_QA 列表依 quiz_page_id → quiz_group_id 分組彙整，並補充 Quiz／Quiz_Group metadata。
    回傳：
    [
      {
        quiz_page_id, tab_name, group_count,
        groups: [
          { quiz_group_id, group_name, unit_name, qa_count, qas: [...] }
        ]
      }
    ]
    """
    if not qas:
        return []

    by_page: dict[str, dict] = OrderedDict()
    all_group_ids: list[int] = []

    for qa in qas:
        page_id = qa.get("quiz_page_id") or ""
        if page_id not in by_page:
            by_page[page_id] = {
                "quiz_page_id": page_id,
                "tab_name": "",
                "groups": OrderedDict(),
            }
        group_id = qa.get("quiz_group_id")
        gkey = str(group_id) if group_id is not None else ""
        groups_dict = by_page[page_id]["groups"]
        if gkey not in groups_dict:
            groups_dict[gkey] = {
                "quiz_group_id": group_id,
                "group_name": "",
                "unit_name": "",
                "qas": [],
            }
            if group_id is not None:
                all_group_ids.append(group_id)
        groups_dict[gkey]["qas"].append(qa)

    tab_name_map = _fetch_tab_name_map(list(by_page.keys()))
    group_meta_map = _fetch_group_meta_map(all_group_ids)

    for page_id, page_data in by_page.items():
        page_data["tab_name"] = tab_name_map.get(page_id, "")
        for grp in page_data["groups"].values():
            gid = grp["quiz_group_id"]
            if gid in group_meta_map:
                meta = group_meta_map[gid]
                grp["group_name"] = meta.get("group_name") or ""
                grp["unit_name"] = meta.get("unit_name") or ""

    result = []
    for page_data in by_page.values():
        groups_list = [
            {
                "quiz_group_id": g["quiz_group_id"],
                "group_name": g["group_name"],
                "unit_name": g["unit_name"],
                "qa_count": len(g["qas"]),
                "qas": g["qas"],
            }
            for g in page_data["groups"].values()
        ]
        result.append({
            "quiz_page_id": page_data["quiz_page_id"],
            "tab_name": page_data["tab_name"],
            "group_count": len(groups_list),
            "groups": groups_list,
        })
    return result
