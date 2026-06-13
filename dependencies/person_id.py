"""
呼叫者身分（person_id）解析：一律由 `Authorization: Bearer <token>` 標頭取得
（POST /v1/auth/login 簽發、POST /v1/auth/refresh 換發）。

2026-06-07 前端已全面改用 Bearer token，query 參數 `person_id` 的過渡 fallback 已移除；
未帶或無效一律 401。
"""

from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException

from utils.auth import identity_from_authorization_header

PERSON_ID_MISSING_DETAIL = (
    "未帶 Authorization: Bearer <token>；請先 POST /v1/auth/login 登入取得 access_token"
)
TOKEN_INVALID_DETAIL = "token 無效或已過期，請重新登入"
# 身分已改為 person_id + college_id；舊版（改版前簽發）token 不含 college_id，一律要求重新登入，
# 避免登入後請求落回「無學校範圍」的不安全行為（例：跨校同 person_id 誤改密碼）。
TOKEN_NO_COLLEGE_DETAIL = "token 缺少學校資訊（college_id），請重新登入"


@dataclass(frozen=True)
class CallerIdentity:
    """呼叫者身分；身分以 person_id + college_id 共同辨識。
    college_id 來自登入學校（舊版 token 可能為 None，代表需重新登入才有學校情境）。"""
    person_id: str
    college_id: Optional[str] = None


def require_identity(
    authorization: Optional[str] = Header(
        None,
        description="Bearer token（POST /v1/auth/login 取得）",
    ),
) -> CallerIdentity:
    if not (authorization or "").strip():
        raise HTTPException(status_code=401, detail=PERSON_ID_MISSING_DETAIL)
    ident = identity_from_authorization_header(authorization)
    if not ident or not ident.get("person_id"):
        raise HTTPException(status_code=401, detail=TOKEN_INVALID_DETAIL)
    if not ident.get("college_id"):
        raise HTTPException(status_code=401, detail=TOKEN_NO_COLLEGE_DETAIL)
    return CallerIdentity(person_id=ident["person_id"], college_id=ident["college_id"])


def require_person_id(
    identity: CallerIdentity = Depends(require_identity),
) -> str:
    return identity.person_id


# 完整身分（person_id + college_id）；需要學校情境的端點（如查／改自己的 User 列）請用此依賴。
CurrentUser = Annotated[CallerIdentity, Depends(require_identity)]
# 僅需 person_id（多數以 course_id／page_id scope 的端點）；向後相容。
PersonId = Annotated[str, Depends(require_person_id)]
