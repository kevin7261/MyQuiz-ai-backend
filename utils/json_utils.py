"""
JSON 序列化相關工具模組。
將 Supabase/DB 回傳值中的 datetime、date 等轉成可 JSON 序列化的型別，避免 API 回應時發生 500 錯誤。
"""

# 引入 date、datetime，用於判斷並轉換日期時間型別
from datetime import date, datetime
# 引入 Any 型別，表示任意型別
from typing import Any

# 將 API 中的 created_at / updated_at 統一為台北時間字串
from utils.datetime_utils import to_taipei_iso


def to_json_safe(obj: Any) -> Any:
    """
    將物件遞迴轉成可 JSON 序列化的型別。
    處理 Supabase/DB 回傳的 datetime、date、dict、list 等，與 rag、exam 等 API 一致。
    """
    # 若為 None，直接回傳 None
    if obj is None:  # None 直接回傳
        return None
    # 若為 datetime 或 date，轉成 ISO 字串
    if isinstance(obj, (datetime, date)):  # 日期時間轉 ISO 字串
        return obj.isoformat()
    # 若為 dict，遞迴處理每個鍵值；created_at / updated_at 轉台北時間
    if isinstance(obj, dict):  # dict 遞迴處理每個值
        return {
            k: to_taipei_iso(v) if k in ("created_at", "updated_at") else to_json_safe(v)
            for k, v in obj.items()
        }
    # 若為 list，遞迴處理每個元素
    if isinstance(obj, list):  # list 遞迴處理每個元素
        return [to_json_safe(v) for v in obj]
    # 若有 keys 方法且非 dict（如 Row 物件），先轉成 dict 再遞迴處理
    if hasattr(obj, "keys") and not isinstance(obj, dict):  # Row 等類似 dict 的物件
        return to_json_safe(dict(obj))
    # 若為基本型別（str、int、float、bool），直接回傳
    if isinstance(obj, (str, int, float, bool)):  # 基本型別直接回傳
        return obj
    return obj  # 其他型別直接回傳
