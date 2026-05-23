"""
JSON 序列化輔助工具。

將 Supabase/DB 回傳值中的 datetime、date 等型別轉成可 JSON 序列化的型別，
避免 FastAPI 序列化時發生 500 錯誤。created_at / updated_at 統一轉為台北時間 ISO 字串。
"""

from datetime import date, datetime
from typing import Any

from utils.datetime_utils import to_taipei_iso


def to_json_safe(obj: Any) -> Any:
    """
    將物件遞迴轉成可 JSON 序列化的型別。

    - datetime / date → 台北時間 ISO 字串
    - dict：created_at / updated_at 欄位統一轉台北時間；其餘欄位遞迴處理
    - list：每個元素遞迴處理
    - 類 dict 物件（Row 等）：先轉 dict 再遞迴
    - 基本型別（str / int / float / bool）及 None：原樣回傳
    """
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return to_taipei_iso(obj)
    if isinstance(obj, dict):
        return {
            k: to_taipei_iso(v) if k in ("created_at", "updated_at") else to_json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    # 相容 postgrest Row 等類 dict 物件（有 keys() 方法但非 dict）
    if hasattr(obj, "keys") and not isinstance(obj, dict):
        return to_json_safe(dict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return obj
