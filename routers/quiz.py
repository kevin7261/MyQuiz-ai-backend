"""
Rag_Quiz 與 Rag_Answer 關聯 API：僅支援 quiz_type=1，用於查看答題結果。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from utils.supabase_client import get_supabase

router = APIRouter(prefix="/quiz", tags=["quiz"])


def _quizzes_with_answers(person_id: Optional[str]):
    """
    固定 quiz_type=1（不看 RAG，只看答題結果）。依條件查 Rag_Quiz 表，再依 rag_quiz_id 查 Rag_Answer 表，回傳每筆 quiz 帶關聯的 answers。
    person_id 未傳則回傳全部。
    """
    supabase = get_supabase()
    q = supabase.table("Rag_Quiz").select("*").eq("quiz_type", 1)
    if person_id is not None and str(person_id).strip():
        q = q.eq("person_id", person_id.strip())
    quiz_resp = q.execute()
    quizzes = quiz_resp.data or []
    if not quizzes:
        return []
    rag_quiz_ids = [q["rag_quiz_id"] for q in quizzes if q.get("rag_quiz_id") is not None]
    if not rag_quiz_ids:
        return [{"quiz": q, "answers": []} for q in quizzes]
    # Rag_Answer 依 rag_quiz_id 關聯（Rag_Quiz 已篩選 quiz_type=1）
    ans_resp = (
        supabase.table("Rag_Answer")
        .select("*")
        .in_("rag_quiz_id", rag_quiz_ids)
        .execute()
    )
    answers = ans_resp.data or []
    answers_by_rag_quiz: dict[int, list[dict]] = {qid: [] for qid in rag_quiz_ids}
    for a in answers:
        qid = a.get("rag_quiz_id")
        if qid is not None:
            answers_by_rag_quiz.setdefault(qid, []).append(a)
    return [
        {"quiz": q, "answers": answers_by_rag_quiz.get(q.get("rag_quiz_id"), [])}
        for q in quizzes
    ]


QUIZ_ANSWERS_RESPONSE_EXAMPLE = {
    "items": [
        {
            "quiz": {
                "rag_quiz_id": 1,
                "rag_id": 0,
                "rag_tab_id": "",
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
                    "rag_answer_id": 101,
                    "rag_quiz_id": 1,
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
            "description": "Rag_Quiz 與關聯的 Rag_Answer 列表",
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
    取得 Rag_Quiz 與 Rag_Answer 的關聯資料（固定 quiz_type=1，不用 RAG，僅用於查看答題結果）。
    回傳 list of { quiz, answers }，每筆 quiz 帶其關聯的 answers（Rag_Answer 表依 rag_quiz_id 關聯）。
    """
    try:
        items = _quizzes_with_answers(person_id)
        return {"items": items, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
