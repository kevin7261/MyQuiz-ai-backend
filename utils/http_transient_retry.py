"""對 httpx／PostgREST 之短暫網路錯誤做少量重試（例如 macOS ReadError errno 35）。"""

import logging
import time
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")
_logger = logging.getLogger(__name__)

# 不含 APIError（4xx 業務／SQL）；僅傳輸層可重試錯誤
_TRANSIENT_HTTPX = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.TimeoutException,
)


def call_with_transient_http_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay_sec: float = 0.08,
) -> T:
    """執行 fn；若為暫時性讀寫／連線錯誤則指數退避重試。"""
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except _TRANSIENT_HTTPX as e:
            last = e
            if i + 1 >= attempts:
                raise
            delay = base_delay_sec * (2**i)
            _logger.warning(
                "暫時性 HTTP 錯誤，%.3fs 後重試 (%d/%d): %s",
                delay,
                i + 1,
                attempts,
                e,
            )
            time.sleep(delay)
    assert last is not None
    raise last
