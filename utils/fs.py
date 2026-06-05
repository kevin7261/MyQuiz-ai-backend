"""檔案系統小工具。"""

from pathlib import Path


def safe_unlink(p: Path) -> None:
    """刪除暫存檔；忽略檔案不存在或刪除失敗。"""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass
