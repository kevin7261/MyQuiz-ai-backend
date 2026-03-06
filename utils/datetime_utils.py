"""日期時間相關工具。"""

from datetime import datetime, timezone


def now_utc_iso() -> str:
    """回傳目前 UTC 時間的 ISO 字串，供 Rag 表 updated_at 使用。"""
    return datetime.now(timezone.utc).isoformat()
