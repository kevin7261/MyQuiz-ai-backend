"""
呼叫者身分（person_id）解析。

優先順序：
1. `Authorization: Bearer <token>` 標頭（POST /v1/auth/login 簽發）— 正式作法
2. query 參數 `person_id` — 過渡期 fallback（前端全面改用 token 後移除）

帶了 Bearer token 但無效／過期 → 401（不會退回 query，避免冒充）。
"""

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Query

from utils.auth import person_id_from_authorization_header

PERSON_ID_MISSING_DETAIL = (
    "請帶 Authorization: Bearer <token>（POST /v1/auth/login 取得）；"
    "過渡期亦可暫用 query 參數 person_id"
)
TOKEN_INVALID_DETAIL = "token 無效或已過期，請重新登入"


def require_person_id(
    authorization: Optional[str] = Header(
        None,
        description="Bearer token（POST /v1/auth/login 取得）",
    ),
    person_id_q: Optional[str] = Query(
        None,
        alias="person_id",
        deprecated=True,
        description="（過渡期）呼叫者 person_id；請改用 Authorization header",
    ),
) -> str:
    if (authorization or "").strip():
        pid = person_id_from_authorization_header(authorization)
        if not pid:
            raise HTTPException(status_code=401, detail=TOKEN_INVALID_DETAIL)
        return pid
    pid = (person_id_q or "").strip()
    if not pid:
        raise HTTPException(status_code=401, detail=PERSON_ID_MISSING_DETAIL)
    return pid


PersonId = Annotated[str, Depends(require_person_id)]
