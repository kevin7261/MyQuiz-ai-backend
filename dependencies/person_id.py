"""所有 API 必填：query 參數 `person_id`。"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Query

PERSON_ID_MISSING_DETAIL = "請傳入 query 參數 person_id"


def require_person_id(
    person_id_q: Optional[str] = Query(
        None,
        alias="person_id",
        description="呼叫者 person_id（必填 query）",
    ),
) -> str:
    pid = (person_id_q or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail=PERSON_ID_MISSING_DETAIL)
    return pid


PersonId = Annotated[str, Depends(require_person_id)]
