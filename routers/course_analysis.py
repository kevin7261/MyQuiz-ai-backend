"""
課程分析 API 模組。
回傳 Exam_Quiz 資料表全部內容。新 schema 中答案欄位（answer_content／answer_critique）
直接內嵌於 Exam_Quiz，不再有獨立的 Exam_Answer 表。
- GET /course-analysis/quizzes：回傳全部 Exam_Quiz，依 exam_tab_id 分群對應 Exam；
  每筆 Exam 含 quizzes（Exam_Quiz 列，answer 內嵌）與 answers（Exam 下所有已作答的摘要）。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from services.exam_queries import all_exam_quizzes, exams_by_tab_ids
from utils.json_utils import to_json_safe

router = APIRouter(prefix="/course-analysis", tags=["course analysis"])


class ListQuizzesResponse(BaseModel):
    """GET /course-analysis/quizzes 回應。格式與 GET /person-analysis/quizzes/{person_id} 一致。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(default=None, description="課程分析不產出，固定為 null")


def _quiz_has_answer(quiz: dict) -> bool:
    """判斷 Exam_Quiz 是否已作答（answer_content 非空）。"""
    return bool((quiz.get("answer_content") or "").strip())


def _synthetic_answer_from_quiz(quiz: dict) -> dict:
    """從 Exam_Quiz 的內嵌欄位構造 answer 摘要（供 answers[] 陣列）。"""
    return {
        "exam_quiz_id": quiz.get("exam_quiz_id"),
        "quiz_answer": quiz.get("answer_content"),
        "answer_critique": quiz.get("answer_critique"),
    }


@router.get("/quizzes", response_model=ListQuizzesResponse)
def list_exam_quizzes(_person_id: PersonId):
    """
    回傳 Exam_Quiz 全部內容。
    exams 陣列：每筆 Exam 含 quizzes（Exam_Quiz）與 answers（已作答的內嵌摘要）。
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

        for row in exam_rows:
            tid = str(row.get("exam_tab_id") or "")
            row_quizzes = quizzes_by_tab.get(tid, [])
            row["quizzes"] = row_quizzes
            row["answers"] = [
                _synthetic_answer_from_quiz(q)
                for q in row_quizzes
                if _quiz_has_answer(q)
            ]

        data = to_json_safe(exam_rows)
        return ListQuizzesResponse(exams=data, count=len(data), weakness_report=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
