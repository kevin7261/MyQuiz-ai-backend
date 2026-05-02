"""
課程分析 API 模組。
回傳 Exam_Quiz 資料表全部內容。新 schema 中答案欄位（answer_content／answer_critique）
直接內嵌於 Exam_Quiz，不再有獨立的 Exam_Answer 表。
- GET /course-analysis/quizzes：回傳全部 Exam_Quiz，依 exam_tab_id 分群對應 Exam；
  每筆 Exam 的題目結構與 GET /exam/tabs 相同（units[]，依 unit_name 分群之 Exam_Quiz，含 enrich／rag 鍵；作答內嵌於各題列）。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from services.exam_queries import (
    all_exam_quizzes,
    exams_by_tab_ids,
    enrich_exam_quizzes_rag_tab_from_units,
    ensure_exam_quiz_rag_id_keys,
    group_exam_quizzes_into_units,
)
from utils.json_utils import to_json_safe

router = APIRouter(prefix="/course-analysis", tags=["course analysis"])


class ListQuizzesResponse(BaseModel):
    """GET /course-analysis/quizzes 回應。exams[] 每筆與 GET /exam/tabs 相同含 units[]（另含 weakness_report 固定 null）。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(default=None, description="課程分析不產出，固定為 null")


@router.get("/quizzes", response_model=ListQuizzesResponse)
def list_exam_quizzes(_person_id: PersonId):
    """
    回傳 Exam_Quiz 全部內容。
    exams 陣列：每筆 Exam 的 units／quizzes 形狀與 GET /exam/tabs 一致。
    weakness_report 固定為 null。
    """
    try:
        quizzes = all_exam_quizzes()

        tab_ids: list[str] = []
        for row in quizzes:
            tid = row.get("exam_tab_id")
            if tid is not None:
                tab_ids.append(str(tid))
        tab_ids = list(dict.fromkeys(tab_ids))

        exam_rows = exams_by_tab_ids(tab_ids)
        quizzes_by_tab: dict[str, list[dict]] = {tid: [] for tid in tab_ids}
        for q in quizzes:
            tid = q.get("exam_tab_id")
            if tid is not None:
                quizzes_by_tab.setdefault(str(tid), []).append(q)

        flat_for_enrich = [qz for tid in tab_ids for qz in quizzes_by_tab.get(tid, [])]
        enrich_exam_quizzes_rag_tab_from_units(flat_for_enrich)
        ensure_exam_quiz_rag_id_keys(flat_for_enrich)

        for row in exam_rows:
            tid = str(row.get("exam_tab_id") or "")
            row["units"] = group_exam_quizzes_into_units(quizzes_by_tab.get(tid, []))

        data = to_json_safe(exam_rows)
        return ListQuizzesResponse(exams=data, count=len(data), weakness_report=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
