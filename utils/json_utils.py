"""JSON 序列化相關工具。"""

from datetime import date, datetime
from typing import Any


def to_json_safe(obj: Any) -> Any:
    """將 Supabase/DB 回傳值轉成可 JSON 序列化的型別（與 rag、exam 一致，避免 500）。"""
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if hasattr(obj, "keys") and not isinstance(obj, dict):
        return to_json_safe(dict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return obj
