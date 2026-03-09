"""
日期時間相關工具模組。
提供 UTC 時間的 ISO 字串，供資料庫 updated_at 等欄位使用。
"""

# 引入 datetime 模組的 datetime 與 timezone，用於取得 UTC 時間
from datetime import datetime, timezone


def now_utc_iso() -> str:
    """
    回傳目前 UTC 時間的 ISO 8601 格式字串。
    供 Rag、Exam 等表的 updated_at 欄位使用。
    """
    # 取得目前 UTC 時間並轉成 ISO 格式字串（如 2025-03-10T12:34:56.123456+00:00）
    return datetime.now(timezone.utc).isoformat()
