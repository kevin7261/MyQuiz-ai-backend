"""quiz_analysis API 必填：query 參數 `quiz_page_id`（試卷識別碼）。"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Query


def require_quiz_page_id(
    quiz_page_id_q: Optional[str] = Query(
        None,
        alias="quiz_page_id",
        description="試卷識別碼（必填 query）",
    ),
) -> str:
    if not quiz_page_id_q or not quiz_page_id_q.strip():
        raise HTTPException(status_code=400, detail="請傳入 query 參數 quiz_page_id")
    return quiz_page_id_q.strip()


QuizPageId = Annotated[str, Depends(require_quiz_page_id)]
