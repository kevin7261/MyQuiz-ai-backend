"""
當下游 HTTP API 回傳可重試的 HTTP 500 時，暫停後重試，預設**直到成功為止**（間隔見 HTTP_500_RETRY_DELAY_SEC）。
供 Supabase（PostgREST / Storage）monkey-patch 與 OpenAI 等呼叫使用。

若需限制次數，可對 call_with_500_retry 傳入 max_retries（正整數）。
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

# 預設 None = 不限制次數，對可重試的 HTTP 500 一直重試到請求成功
HTTP_500_MAX_RETRIES: int | None = None
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
    max_retries: int | None = HTTP_500_MAX_RETRIES,
    delay_sec: float = HTTP_500_RETRY_DELAY_SEC,
) -> T:
    """
    執行 fn；若拋出可重試的 HTTP 500，則延遲後重試。

    max_retries 為 None 時不限制次數，直到成功或拋出非可重試錯誤。
    為正整數時，失敗後最多再重試 max_retries 次（與舊版「最多 5 次再試」語意相同）。
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if not is_retryable_http_500(e):
                raise
            if max_retries is not None and attempt >= max_retries:
                raise
            attempt += 1
            time.sleep(delay_sec)


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
