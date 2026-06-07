"""
Supabase 連線模組。

從環境變數讀取 SUPABASE_URL 與 Key，建立並快取 Supabase 客戶端（單例）。
後端預設使用 service_role key，可略過 RLS；需依使用者權限時改用 anon key + 使用者 JWT。

Key 優先順序（use_service_role=True）：
  SUPABASE_SERVICE_ROLE_KEY → SUPABASE_SECRET_KEY → SUPABASE_ANON_KEY（退回並警告）

另外在首次建立客戶端時 patch postgrest 的 .execute() 與 storage3 的 ._request()，
對暫時性連線錯誤自動重試：單例客戶端的 httpx keep-alive 連線閒置後會被 Supabase 端
關閉，下一個請求重用死連線即拋 RemoteProtocolError／ReadError（症狀：偶發 500，
重新整理變 200）。
"""

import functools
import io
import os
import warnings
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from supabase import Client

# 單例快取：service_role 與 anon 各一份，避免重複建立連線
_client_service: Optional["Client"] = None
_client_anon: Optional["Client"] = None

# postgrest／storage3 重試 patch 只做一次
_transient_retry_patched = False


def _patch_supabase_transient_retry() -> None:
    """
    讓所有 supabase.table(...).execute()（postgrest）與
    storage.download/upload/update/remove/list（storage3）自動套用
    call_with_transient_http_retry。
    """
    global _transient_retry_patched
    if _transient_retry_patched:
        return

    from postgrest._sync import request_builder as _rb
    from storage3._sync.file_api import SyncBucketActionsMixin as _bucket

    from utils.retry import call_with_transient_http_retry

    # --- postgrest：只包「實際送出 HTTP 請求」的類別。
    # SyncMaybeSingleRequestBuilder 內部委派給 SyncSingleRequestBuilder，
    # 一併包裝會變成巢狀重試（4×4 次），故略過。
    for cls in (
        _rb.SyncQueryRequestBuilder,
        _rb.SyncSingleRequestBuilder,
        _rb.SyncExplainRequestBuilder,
    ):
        original = cls.execute

        @functools.wraps(original)
        def execute_with_retry(self, _original=original):
            return call_with_transient_http_retry(lambda: _original(self))

        cls.execute = execute_with_retry

    # --- storage3：所有 bucket 操作（download/upload/update/remove/list…）
    # 都經過 SyncBucketActionsMixin._request，包這一處即可全部覆蓋。
    _original_request = _bucket._request

    @functools.wraps(_original_request)
    def request_with_retry(self, *args, files=None, **kwargs):
        def _once():
            return _original_request(self, *args, files=files, **kwargs)

        # 上傳內容為檔案串流（BufferedReader）時不重試：
        # 串流可能已被部分讀取，重送會送出壞資料。本專案上傳皆傳 bytes，可安全重送。
        if isinstance(files, dict):
            f = files.get("file")
            if isinstance(f, tuple) and len(f) >= 2 and isinstance(f[1], io.BufferedReader):
                return _once()
        return call_with_transient_http_retry(_once)

    _bucket._request = request_with_retry

    _transient_retry_patched = True


def get_supabase(use_service_role: bool = True):
    """
    取得 Supabase 客戶端（單例）。

    use_service_role=True：使用 service_role key（後端管理用，可略過 RLS）。
    use_service_role=False：使用 anon key（前端限制權限情境）。
    """
    global _client_service, _client_anon

    if use_service_role:
        if _client_service is not None:
            return _client_service
        key_name = "SUPABASE_SERVICE_ROLE_KEY"
    else:
        if _client_anon is not None:
            return _client_anon
        key_name = "SUPABASE_ANON_KEY"

    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get(key_name) or "").strip()

    # service_role 未設定時，嘗試新版 Secret key（sb_secret_...）
    if use_service_role and not key:
        key = (os.environ.get("SUPABASE_SECRET_KEY") or "").strip()

    # 仍無 key 時退回 anon key，並警告（RLS 可能擋寫入操作）
    if use_service_role and not key:
        warnings.warn(
            "SUPABASE_SERVICE_ROLE_KEY / SUPABASE_SECRET_KEY 未設定，暫時改用 SUPABASE_ANON_KEY。"
            "請到 Supabase → Project Settings → API 取得 service_role 或 Secret key。"
        )
        key = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()

    if not url or not key:
        raise RuntimeError(
            "Missing Supabase config: 請在 .env 設定 SUPABASE_URL 與 "
            "SUPABASE_SERVICE_ROLE_KEY（或至少 SUPABASE_ANON_KEY）"
        )

    # URL 格式驗證：防止空格、換行或填錯值導致連線失敗
    if not url.startswith("https://") or ".supabase.co" not in url:
        raise RuntimeError(
            f"SUPABASE_URL 格式錯誤（目前值：{url!r}），"
            "應為 https://xxxxx.supabase.co（請檢查 .env，勿有多餘空格或換行）"
        )

    from supabase import create_client

    _patch_supabase_transient_retry()
    client = create_client(url, key)
    if use_service_role:
        _client_service = client
    else:
        _client_anon = client
    return client
