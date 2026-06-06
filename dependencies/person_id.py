"""
呼叫者身分（person_id）解析：一律由 `Authorization: Bearer <token>` 標頭取得
（POST /v1/auth/login 簽發、POST /v1/auth/refresh 換發）。

2026-06-07 前端已全面改用 Bearer token，query 參數 `person_id` 的過渡 fallback 已移除；
未帶或無效一律 401。
"""

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException

from utils.auth import person_id_from_authorization_header

PERSON_ID_MISSING_DETAIL = (
    "未帶 Authorization: Bearer <token>；請先 POST /v1/auth/login 登入取得 access_token"
)
TOKEN_INVALID_DETAIL = "token 無效或已過期，請重新登入"


def require_person_id(
    authorization: Optional[str] = Header(
        None,
        description="Bearer token（POST /v1/auth/login 取得）",
    ),
) -> str:
    if not (authorization or "").strip():
        raise HTTPException(status_code=401, detail=PERSON_ID_MISSING_DETAIL)
    pid = person_id_from_authorization_header(authorization)
    if not pid:
        raise HTTPException(status_code=401, detail=TOKEN_INVALID_DETAIL)
    return pid


PersonId = Annotated[str, Depends(require_person_id)]
