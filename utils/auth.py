"""
Bearer token 簽發與驗證（HMAC-SHA256 自簽 token，無外部依賴）。

- POST /v1/auth/login 成功後簽發 access_token
- dependencies.person_id 於每次請求驗證 Authorization: Bearer <token>
- 密鑰來源：env AUTH_TOKEN_SECRET（建議於 .env／部署平台設定）；
  未設定時退回 SUPABASE_SERVICE_ROLE_KEY 衍生值。兩者皆未設定時 fail-closed
  （直接 raise，不再退回寫死的開發密鑰，避免正式環境漏設時可被已知密鑰偽造 token）。

token 格式：base64url(JSON payload).base64url(HMAC-SHA256 簽章)
payload：{"sub": person_id, "iat": 簽發時間, "exp": 到期時間}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

AUTH_TOKEN_SECRET_ENV = "AUTH_TOKEN_SECRET"
# 預設 30 天；可用 env AUTH_TOKEN_TTL_SECONDS 覆寫
DEFAULT_TOKEN_TTL_SECONDS = 30 * 24 * 3600


def _secret() -> bytes:
    s = (os.environ.get(AUTH_TOKEN_SECRET_ENV) or "").strip()
    if not s:
        s = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not s:
        # fail-closed：過去退回寫死的開發密鑰，正式環境若漏設兩個 env，
        # 任何人都能用該已知密鑰自簽 token 繞過認證。改為直接拒絕簽發／驗證。
        raise RuntimeError(
            "未設定 AUTH_TOKEN_SECRET 或 SUPABASE_SERVICE_ROLE_KEY，無法簽發／驗證 token。"
            "請於環境變數設定 AUTH_TOKEN_SECRET（建議）或 SUPABASE_SERVICE_ROLE_KEY。"
        )
    return hashlib.sha256(s.encode("utf-8")).digest()


def token_ttl_seconds() -> int:
    raw = (os.environ.get("AUTH_TOKEN_TTL_SECONDS") or "").strip()
    try:
        ttl = int(raw)
        if ttl > 0:
            return ttl
    except ValueError:
        pass
    return DEFAULT_TOKEN_TTL_SECONDS


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _sign(payload_b64: str) -> str:
    sig = hmac.new(_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(sig)


def issue_token(person_id: str) -> str:
    """為登入成功的 person_id 簽發 Bearer token。"""
    now = int(time.time())
    payload = {"sub": str(person_id), "iat": now, "exp": now + token_ttl_seconds()}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: str) -> Optional[str]:
    """驗證 token；成功回傳 person_id，失敗（格式錯誤／簽章不符／過期）回 None。"""
    try:
        payload_b64, sig = (token or "").strip().split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload_b64)):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp") or 0) < int(time.time()):
            return None
        sub = str(payload.get("sub") or "").strip()
        return sub or None
    except Exception:
        return None


def person_id_from_authorization_header(authorization: Optional[str]) -> Optional[str]:
    """從 `Authorization: Bearer <token>` 標頭解出 person_id；無標頭或非 Bearer 回 None。"""
    value = (authorization or "").strip()
    if not value.lower().startswith("bearer "):
        return None
    return verify_token(value[7:].strip())
