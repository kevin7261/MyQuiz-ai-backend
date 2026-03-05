"""
Quiz 與 Answer 關聯 API（非 RAG）：僅支援 quiz_type=1，用於查看答題結果。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from utils.supabase_client import get_supabase

router = APIRouter(prefix="/quiz", tags=["quiz"])


def _quizzes_with_answers(person_id: Optional[str]):
    """
    固定 quiz_type=1（不看 RAG，只看答題結果）。依條件查 Quiz 表，再依 quiz_id 查 Answer 表，回傳每筆 quiz 帶關聯的 answers。
    person_id 未傳則回傳全部。
    """
    supabase = get_supabase()
    q = supabase.table("Quiz").select("*").eq("quiz_type", 1)
    if person_id is not None and str(person_id).strip():
        q = q.eq("person_id", person_id.strip())
    quiz_resp = q.execute()
    quizzes = quiz_resp.data or []
    if not quizzes:
        return []
    quiz_ids = [q["quiz_id"] for q in quizzes if q.get("quiz_id") is not None]
    if not quiz_ids:
        return [{"quiz": q, "answers": []} for q in quizzes]
    # Answer 依 quiz_id 關聯（Quiz 已篩選 quiz_type=1）
    ans_resp = (
        supabase.table("Answer")
        .select("*")
        .in_("quiz_id", quiz_ids)
        .execute()
    )
    answers = ans_resp.data or []
    answers_by_quiz: dict[int, list[dict]] = {qid: [] for qid in quiz_ids}
    for a in answers:
        qid = a.get("quiz_id")
        if qid is not None:
            answers_by_quiz.setdefault(qid, []).append(a)
    return [
        {"quiz": q, "answers": answers_by_quiz.get(q.get("quiz_id"), [])}
        for q in quizzes
    ]


QUIZ_ANSWERS_RESPONSE_EXAMPLE = {
    "items": [
        {
            "quiz": {
                "quiz_id": 1,
                "rag_id": 0,
                "file_id": "",
                "person_id": "user_abc",
                "course_name": "經濟學",
                "quiz_content": "請說明供給與需求的關係。",
                "quiz_hint": "可從價格機制談起。",
                "reference_answer": "供給增加時價格下降…",
                "quiz_type": 1,
                "quiz_level": 0,
                "created_at": "2025-03-01T10:00:00",
                "updated_at": "2025-03-01T10:00:00",
            },
            "answers": [
                {
                    "answer_id": 101,
                    "quiz_id": 1,
                    "person_id": "user_abc",
                    "student_answer": "供給增加會讓價格下降，需求增加會讓價格上升。",
                    "answer_grade": 8,
                    "answer_feedback_metadata": "{\"score\": 8, \"level\": \"良好\"}",
                    "created_at": "2025-03-02T14:00:00",
                    "updated_at": "2025-03-02T14:00:00",
                }
            ],
        }
    ],
    "count": 1,
}


@router.get(
    "/quiz-answers",
    responses={
        200: {
            "description": "Quiz 與關聯的 Answer 列表",
            "content": {
                "application/json": {
                    "example": QUIZ_ANSWERS_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
async def get_quiz_with_answers(
    person_id: Optional[str] = Query(None, description="選填，篩選 person_id；未傳則回傳全部"),
):
    """
    取得 Quiz 與 Answer 的關聯資料（固定 quiz_type=1，不用 RAG，僅用於查看答題結果）。
    回傳 list of { quiz, answers }，每筆 quiz 帶其關聯的 answers（Answer 表依 quiz_id 關聯）。
    """
    try:
        items = _quizzes_with_answers(person_id)
        return {"items": items, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
