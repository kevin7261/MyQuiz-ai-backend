"""Bank 專屬：自 Bank_Unit 列取逐字稿（自 utils.rag_stem.transcript_from_row 複製，與 rag 無關）。"""

from __future__ import annotations


def transcript_from_row(row: dict | None) -> str:
    """Bank_Unit 逐字稿欄位（新欄 transcript；舊欄 transcription 向下相容）。"""
    if not row:
        return ""
    return (row.get("transcript") or row.get("transcription") or "").strip()
