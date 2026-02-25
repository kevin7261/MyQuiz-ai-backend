"""
Supabase 連線：從環境變數讀取，供課程／使用者等 API 使用。
後端預設使用 service_role key（可略過 RLS）；需依登入使用者權限時改用 anon key + 使用者 JWT。
"""
import os
from typing import Optional

_client_service: Optional["SupabaseClient"] = None
_client_anon: Optional["SupabaseClient"] = None


def get_supabase(use_service_role: bool = True):
    """取得 Supabase client（單例）。預設用 service_role（後端管理／略過 RLS）。"""
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

    # 若 service_role 未設定，可改用新版 Secret key（Supabase 後台「Secret keys」區的 sb_secret_...）
    if use_service_role and not key:
        key = (os.environ.get("SUPABASE_SECRET_KEY") or "").strip()
    if use_service_role and not key:
        import warnings
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


def get_supabase_anon():
    """取得使用 anon key 的 client（會受 RLS 限制，適合依使用者 token 查資料）。"""
    return get_supabase(use_service_role=False)
