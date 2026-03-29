"""
Supabase 連線模組。
從環境變數讀取 SUPABASE_URL 與 Key，建立 Supabase 客戶端，供課程、使用者等 API 使用。
後端預設使用 service_role key（可略過 RLS）；需依登入使用者權限時改用 anon key + 使用者 JWT。
"""

# 引入 os 模組以讀取環境變數
import os
# 引入 Optional 型別
from typing import Optional

# 全域變數：service_role 客戶端（單例）
_client_service: Optional["SupabaseClient"] = None
# 全域變數：anon 客戶端（單例）
_client_anon: Optional["SupabaseClient"] = None


def get_supabase(use_service_role: bool = True):
    """
    取得 Supabase 客戶端（單例）。
    use_service_role=True 時使用 service_role key（後端管理、略過 RLS）；
    否則使用 anon key。
    """
    # 宣告使用全域變數
    global _client_service, _client_anon
    # 依據是否使用 service_role 選擇對應客戶端
    if use_service_role:
        # 若 service_role 客戶端已存在，直接回傳
        if _client_service is not None:
            return _client_service
        key_name = "SUPABASE_SERVICE_ROLE_KEY"
    else:
        # 若 anon 客戶端已存在，直接回傳
        if _client_anon is not None:
            return _client_anon
        key_name = "SUPABASE_ANON_KEY"

    # 從環境變數取得 URL
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    # 從環境變數取得對應的 Key
    key = (os.environ.get(key_name) or "").strip()

    # 若 service_role 未設定，可改用 Secret key（Supabase 後台「Secret keys」區的 sb_secret_...）
    if use_service_role and not key:
        key = (os.environ.get("SUPABASE_SECRET_KEY") or "").strip()
    # 若仍無 key，暫時改用 anon key 並發出警告
    if use_service_role and not key:
        import warnings  # 動態引入 warnings
        warnings.warn(  # 發出警告
            "SUPABASE_SERVICE_ROLE_KEY / SUPABASE_SECRET_KEY 未設定，暫時改用 SUPABASE_ANON_KEY。"
            "請到 Supabase → Project Settings → API 取得 service_role 或 Secret key。"
        )
        key = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()

    # 若 URL 或 Key 缺失，拋出 RuntimeError
    if not url or not key:
        raise RuntimeError(
            "Missing Supabase config: 請在 .env 設定 SUPABASE_URL 與 "
            "SUPABASE_SERVICE_ROLE_KEY（或至少 SUPABASE_ANON_KEY）"
        )
    # 檢查 URL 格式（應為 https 且包含 .supabase.co）
    if not url.startswith("https://") or ".supabase.co" not in url:
        raise RuntimeError(
            f"SUPABASE_URL 格式錯誤（目前值：{url!r}），"
            "應為 https://xxxxx.supabase.co（請檢查 .env，勿有多餘空格或換行）"
        )

    # 動態引入 create_client（避免啟動時就依賴 supabase）
    from supabase import create_client
    # 建立客戶端
    client = create_client(url, key)
    # 依據 use_service_role 存入對應的全域變數
    if use_service_role:
        _client_service = client
    else:
        _client_anon = client
    return client
