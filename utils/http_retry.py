"""
當下游 HTTP API 回傳 status 500 時，暫停後重試（預設間隔 5 秒、最多重試 5 次）。
供 Supabase（PostgREST / Storage）monkey-patch 與 OpenAI 等呼叫使用。
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

# 第一次請求失敗後再重試的次數（共最多 1 + 5 次請求）
HTTP_500_MAX_RETRIES = 5
HTTP_500_RETRY_DELAY_SEC = 5.0

T = TypeVar("T")

_patches_installed = False


def is_retryable_http_500(exc: BaseException) -> bool:
    """若為 HTTP 500（或等價錯誤碼）則可重試。"""
    try:
        from postgrest.exceptions import APIError as PostgrestAPIError

        if isinstance(exc, PostgrestAPIError):
            c = exc.code
            return c == 500 or str(c) == "500"
    except ImportError:
        pass

    try:
        from storage3.exceptions import StorageApiError

        if isinstance(exc, StorageApiError):
            s = exc.status
            return s == 500 or str(s) == "500"
    except ImportError:
        pass

    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 500
    except ImportError:
        pass

    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError):
            return exc.status_code == 500
    except ImportError:
        pass

    sc = getattr(exc, "status_code", None)
    if sc == 500:
        return True
    st = getattr(exc, "status", None)
    if st == 500 or str(st) == "500":
        return True
    return False


def call_with_500_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = HTTP_500_MAX_RETRIES,
    delay_sec: float = HTTP_500_RETRY_DELAY_SEC,
) -> T:
    """
    執行 fn；若拋出可重試的 HTTP 500，則延遲後重試，最多重試 max_retries 次。
    """
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if not is_retryable_http_500(e) or attempt >= max_retries:
                raise
            time.sleep(delay_sec)
    assert last is not None
    raise last


def install_supabase_500_retry_patches() -> None:
    """對 postgrest 的 execute 與 storage3 的 _request 安裝 500 重試（僅執行一次）。"""
    global _patches_installed
    if _patches_installed:
        return

    from postgrest._sync import request_builder as rb

    def _wrap(orig: Callable) -> Callable:
        def wrapped(self):  # type: ignore[no-untyped-def]
            return call_with_500_retry(lambda: orig(self))

        return wrapped

    rb.SyncQueryRequestBuilder.execute = _wrap(rb.SyncQueryRequestBuilder.execute)  # type: ignore[method-assign]
    rb.SyncSingleRequestBuilder.execute = _wrap(rb.SyncSingleRequestBuilder.execute)  # type: ignore[method-assign]
    rb.SyncExplainRequestBuilder.execute = _wrap(rb.SyncExplainRequestBuilder.execute)  # type: ignore[method-assign]
    rb.SyncMaybeSingleRequestBuilder.execute = _wrap(rb.SyncMaybeSingleRequestBuilder.execute)  # type: ignore[method-assign]

    from storage3._sync import file_api as fa

    _orig_req = fa.SyncBucketActionsMixin._request

    def _patched_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return call_with_500_retry(lambda: _orig_req(self, *args, **kwargs))

    fa.SyncBucketActionsMixin._request = _patched_request  # type: ignore[method-assign]

    _patches_installed = True
