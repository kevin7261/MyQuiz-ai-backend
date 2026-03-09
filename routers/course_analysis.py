"""
課程分析 API 模組。
回傳 Exam_Quiz 資料表全部內容，格式與 GET /person-analysis/quizzes/{person_id} 一致。
- GET /course-analysis/quizzes：回傳格式同 List Quizzes By Person：exams 陣列，每筆 Exam 含表欄位及 quizzes（每題帶 answers）、頂層 answers；count、weakness_report（課程分析固定為 null）。
"""

# 引入 Optional 型別
from typing import Optional

# 引入 FastAPI 的 APIRouter、HTTPException
from fastapi import APIRouter, HTTPException
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 從 exam 模組引入共用查詢函數
from routers.exam import _all_exam_quizzes, _answers_by_exam_quiz_ids, _exams_by_ids
# 引入 to_json_safe 將 datetime 等轉成可 JSON 序列化
from utils.json_utils import to_json_safe

# 建立路由，前綴 /course-analysis
router = APIRouter(prefix="/course-analysis", tags=["course analysis"])


class ListQuizzesResponse(BaseModel):
    """GET /course-analysis/quizzes 回應。格式與 GET /person-analysis/quizzes/{person_id} 一致。"""
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(default=None, description="課程分析不產出，固定為 null")


@router.get("/quizzes", response_model=ListQuizzesResponse)
def list_exam_quizzes():
    """
    回傳 Exam_Quiz 全部內容，格式與 List Quizzes By Person 相同。
    exams 陣列，每筆含 quizzes、answers；每題 quiz 帶關聯的 answers（Exam_Answer）。
    weakness_report 固定為 null。
    """
    try:
        # 查詢 Exam_Quiz 全部筆數
        quizzes = _all_exam_quizzes()
        quiz_ids: list[int] = []
        for row in quizzes:
            qid = row.get("exam_quiz_id")
            if qid is not None:
                try:
                    quiz_ids.append(int(qid))
                except (TypeError, ValueError):
                    pass
        quiz_ids = list(dict.fromkeys(quiz_ids))
        # 依 exam_quiz_id 查詢 answers
        answers_by_quiz = _answers_by_exam_quiz_ids(quiz_ids)
        for quiz in quizzes:
            qid = quiz.get("exam_quiz_id")
            qid_int = int(qid) if qid is not None else None
            quiz["answers"] = (answers_by_quiz.get(qid_int, []) or []) if qid_int is not None else []
        exam_ids: list[int] = []
        for row in quizzes:
            eid = row.get("exam_id")
            if eid is not None:
                try:
                    exam_ids.append(int(eid))
                except (TypeError, ValueError):
                    pass
        exam_ids = list(dict.fromkeys(exam_ids))
        exam_rows = _exams_by_ids(exam_ids)
        quizzes_by_exam: dict[int, list[dict]] = {eid: [] for eid in exam_ids}
        for q in quizzes:
            eid = q.get("exam_id")
            if eid is not None:
                try:
                    quizzes_by_exam.setdefault(int(eid), []).append(q)
                except (TypeError, ValueError):
                    pass
        for row in exam_rows:
            eid = row.get("exam_id")
            eid_int = int(eid) if eid is not None else None
            row_quizzes = quizzes_by_exam.get(eid_int, []) if eid_int is not None else []
            row["quizzes"] = row_quizzes
            row["answers"] = []
            for q in row_quizzes:
                row["answers"].extend(q.get("answers") or [])
        data = to_json_safe(exam_rows)
        return ListQuizzesResponse(exams=data, count=len(data), weakness_report=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
