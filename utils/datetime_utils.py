"""
日期時間相關工具模組。
以 Asia/Taipei 的 ISO 8601 字串寫入資料庫 updated_at / created_at，並供 API 將欄位轉成台北時間。
"""

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

# 台北時區（含 DST 歷史由 IANA 資料處理）
TAIPEI_TZ = ZoneInfo("Asia/Taipei")


def now_taipei_iso() -> str:
    """
    回傳目前台北時間的 ISO 8601 字串（含 +08:00 等偏移）。
    供 Rag、Exam 等表的 updated_at、created_at 寫入與回傳一致。
    """
    return datetime.now(TAIPEI_TZ).isoformat()


def to_taipei_iso(value: Any) -> Any:
    """
    將資料庫或 Supabase 回傳的時間轉成台北時區 ISO 字串。
    支援 None、datetime、date、ISO 字串（含尾端 Z）；無法解析的字串原樣回傳。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TAIPEI_TZ).isoformat()
    if isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time(), tzinfo=TAIPEI_TZ)
        return dt.isoformat()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TAIPEI_TZ).isoformat()
    return value
