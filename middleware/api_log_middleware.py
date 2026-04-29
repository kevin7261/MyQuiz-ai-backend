"""
每次「實際業務」API 請求寫入 public.Log：person_id、api（路徑 URL）、api_metadata（api、method、parameters）。

不記錄 OPTIONS／HEAD：瀏覽器跨域會先送 CORS preflight（OPTIONS），沒有 JSON body，
若一併記錄會變成「前端呼叫 1 次卻出現 2 筆 log、且 parameters 只有 person_id」。
寫入失敗不影響原請求回應。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_logger = logging.getLogger(__name__)

# 不寫入 Log 的 HTTP 方法（CORS preflight、探測用）
_SKIP_HTTP_METHODS = frozenset({"OPTIONS", "HEAD"})

# 不寫入 Log 的路徑（Swagger／OpenAPI）
_SKIP_PATH_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
)

# 不寫入 Log 的「方法 + 路徑」（精確比對 path，不含 query）
_SKIP_LOG_ROUTES = frozenset({
    ("GET", "/system-settings/course-name"),
})

# parameters 內敏感欄位遮罩
_REDACT_SUBSTRINGS = ("password", "secret", "token", "api_key", "apikey", "authorization")
_REDACT_EXACT = frozenset(
    {"password", "llm_api_key", "llmapikey", "deepgram_api_key", "openai_api_key"}
)


def _should_skip_path(path: str) -> bool:
    p = path or ""
    return any(p == x or p.startswith(x + "/") for x in _SKIP_PATH_PREFIXES)


def _should_skip_route(method: str, path: str) -> bool:
    return ((method or "").upper(), path or "") in _SKIP_LOG_ROUTES


def _stringify_param_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _maybe_redact_key(key: str, value_str: str) -> str:
    k = (key or "").lower().replace("-", "_")
    if k in _REDACT_EXACT:
        return "***"
    for sub in _REDACT_SUBSTRINGS:
        if sub in k:
            return "***"
    return value_str


def _build_parameters(query_params: dict[str, str], body_flat: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in query_params.items():
        out[k] = _maybe_redact_key(k, _stringify_param_value(v))
    for k, v in body_flat.items():
        if k in out:
            continue
        out[k] = _maybe_redact_key(k, _stringify_param_value(v))
    return out


def _insert_log_row(*, person_id: str, api: str, api_metadata: dict[str, Any]) -> None:
    from utils.datetime_utils import now_taipei_iso
    from utils.supabase_client import get_supabase

    ts = now_taipei_iso()
    supabase = get_supabase()
    supabase.table("Log").insert(
        {
            "person_id": (person_id or "")[:255],
            "api": (api or "")[:255],
            "api_metadata": api_metadata,
            "created_at": ts,
            "updated_at": ts,
        }
    ).execute()


async def _log_request_async(
    *,
    person_id: str,
    api_url: str,
    method: str,
    parameters: dict[str, str],
) -> None:
    api_url = (api_url or "")[:255]
    method_l = (method or "get").lower()
    api_metadata = {
        "api": api_url,
        "method": method_l,
        "parameters": parameters,
    }
    try:
        await asyncio.to_thread(
            _insert_log_row,
            person_id=person_id,
            api=api_url,
            api_metadata=api_metadata,
        )
    except Exception:
        _logger.exception("寫入 Log 表失敗（已忽略，不影響 API）")


class APILogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.scope.get("type") != "http":
            return await call_next(request)

        path = request.url.path
        if _should_skip_path(path):
            return await call_next(request)

        method_upper = request.method.upper()
        if method_upper in _SKIP_HTTP_METHODS:
            return await call_next(request)
        if _should_skip_route(method_upper, path):
            return await call_next(request)

        query_params = dict(request.query_params)
        person_id = (query_params.get("person_id") or "").strip()
        qs = request.url.query
        api_url = f"{path}?{qs}" if qs else path
        api_url = api_url[:255]

        method = request.method
        body_flat: dict[str, Any] = {}
        body_bytes = b""
        content_type = (request.headers.get("content-type") or "").lower()

        if method in ("POST", "PUT", "PATCH", "DELETE") and "application/json" in content_type:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    parsed = json.loads(body_bytes.decode("utf-8"))
                    if isinstance(parsed, dict):
                        body_flat = parsed
                    else:
                        body_flat = {"_json": parsed}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body_flat = {
                        "_raw": body_bytes[:2000].decode("utf-8", errors="replace"),
                    }
            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}

            request = Request(request.scope, receive)
        elif method in ("POST", "PUT", "PATCH") and "multipart/form-data" in content_type:
            body_flat = {"_body": "multipart/form-data"}

        parameters = _build_parameters(query_params, body_flat)

        await _log_request_async(
            person_id=person_id,
            api_url=api_url,
            method=method,
            parameters=parameters,
        )

        return await call_next(request)
