"""所有 RAG API 必填：query 參數 `course_id`。"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Query

COURSE_ID_MISSING_DETAIL = "請傳入 query 參數 course_id"


def require_course_id(
    course_id_q: Optional[int] = Query(
        None,
        alias="course_id",
        description="目前課程 ID（必填 query）",
    ),
) -> int:
    if course_id_q is None:
        raise HTTPException(status_code=400, detail=COURSE_ID_MISSING_DETAIL)
    return int(course_id_q)


CourseId = Annotated[int, Depends(require_course_id)]
