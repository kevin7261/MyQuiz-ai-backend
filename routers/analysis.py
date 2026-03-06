"""
分析 API：依 person_id 查詢 Exam_Quiz / Exam_Answer 等分析用資料。
- GET /analysis/quizzes-by-person/{person_id}：依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆帶關聯的 Exam_Answer。
  回傳格式與 GET /rag/rags、GET /exam/exams 的題目答案內容一致（每筆 quiz 含 quiz_content、quiz_hint、reference_answer、quiz_metadata，answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
"""

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Path as PathParam
from pydantic import BaseModel

from routers.exam import _answers_by_exam_quiz_ids, _quizzes_by_person_id

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _to_json_safe(obj: Any) -> Any:
    """將 Supabase/DB 回傳值轉成可 JSON 序列化的型別（與 rag、exam 一致，避免 500）。"""
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "keys") and not isinstance(obj, dict):
        return _to_json_safe(dict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return obj


class ListQuizzesByPersonResponse(BaseModel):
    """GET /analysis/quizzes-by-person/{person_id} 回應：格式同 rag/exam 的題目答案，每筆 quiz 帶 answers。"""
    quizzes: list[dict]
    count: int


@router.get("/quizzes-by-person/{person_id}", response_model=ListQuizzesByPersonResponse)
def list_quizzes_by_person(
    person_id: str = PathParam(..., description="要查詢的 person_id"),
):
    """
    依 person_id 取得該使用者在 Exam_Quiz 的所有資料，每筆 quiz 帶關聯的 Exam_Answer（answers）。
    回傳題目／答案的 JSON 結構與 GET /rag/rags、GET /exam/exams 一致（quiz_content、quiz_hint、reference_answer、quiz_metadata；answers 含 student_answer、answer_grade、answer_feedback_metadata、answer_metadata 等）。
    """
    try:
        quizzes = _quizzes_by_person_id(person_id)
        quiz_ids = []
        for row in quizzes:
            qid = row.get("exam_quiz_id")
            if qid is not None:
                try:
                    quiz_ids.append(int(qid))
                except (TypeError, ValueError):
                    pass
        quiz_ids = list(dict.fromkeys(quiz_ids))
        answers_by_quiz = _answers_by_exam_quiz_ids(quiz_ids)
        for quiz in quizzes:
            qid = quiz.get("exam_quiz_id")
            qid_int = int(qid) if qid is not None else None
            quiz["answers"] = answers_by_quiz.get(qid_int, []) if qid_int is not None else []
        # 與 rag、exam 一致：轉成可 JSON 序列化（datetime 等轉成 ISO 字串）
        data = _to_json_safe(quizzes)
        return ListQuizzesByPersonResponse(quizzes=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
