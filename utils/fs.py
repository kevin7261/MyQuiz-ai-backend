"""檔案系統小工具。"""

from pathlib import Path
from typing import Optional


def safe_unlink(p: Path) -> None:
    """刪除暫存檔；忽略檔案不存在或刪除失敗。"""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def safe_extract_path(root: Path, member_name: str) -> Optional[Path]:
    """計算 ZIP 成員解壓後的目標路徑，並確保其落在 root 之內。

    防止路徑穿越：``decoded.replace("..", "_")`` 無法擋下絕對路徑成員——
    ``root / "/etc/x"`` 在 pathlib 會被絕對路徑覆蓋成 ``/etc/x``。此處改以
    resolve 後的包含關係判斷：成員名含 ``..`` 逃逸或為絕對路徑而落在 root 之外時，
    回傳 None（呼叫端應略過該成員）。合法（純相對、留在 root 內）的成員行為不變。
    """
    root_resolved = root.resolve()
    target = (root / member_name).resolve()
    if target == root_resolved or root_resolved in target.parents:
        return target
    return None
