"""
Supabase 連線模組。

從環境變數讀取 SUPABASE_URL 與 Key，建立並快取 Supabase 客戶端（單例）。
後端預設使用 service_role key，可略過 RLS；需依使用者權限時改用 anon key + 使用者 JWT。

Key 優先順序（use_service_role=True）：
  SUPABASE_SERVICE_ROLE_KEY → SUPABASE_SECRET_KEY → SUPABASE_ANON_KEY（退回並警告）
"""

import os
import warnings
from typing import Optional

# 單例快取：service_role 與 anon 各一份，避免重複建立連線
_client_service: Optional["SupabaseClient"] = None
_client_anon: Optional["SupabaseClient"] = None


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

    client = create_client(url, key)
    if use_service_role:
        _client_service = client
    else:
        _client_anon = client
    return client
