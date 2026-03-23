"""
RAG 相關共用邏輯模組。
提供由 rag_id 查詢 Rag 表並取得 stem、rag_zip_tab_id 的函數。
"""

# 引入 FastAPI 的 HTTPException，用於拋出 404、400 等錯誤
from pathlib import Path

from fastapi import HTTPException


def get_rag_stem_from_rag_id(supabase, rag_id: int, include_row: bool = False):
    """
    由 rag_id 查詢 Rag 表，回傳 stem（RAG ZIP 的 tab_id）與 rag_zip_tab_id（通常為 {stem}_rag）。
    include_row=True 時回傳 (row, stem, rag_zip_tab_id)；否則回傳 (stem, rag_zip_tab_id)。
    """
    # 依據 include_row 決定查詢欄位：需要完整 row 時多查 rag_tab_id、system_prompt_instruction、person_id、rag_id
    # include_row 時多查 rag_tab_id、person_id 等
    select_cols = "rag_tab_id, system_prompt_instruction, person_id, rag_id, rag_metadata" if include_row else "rag_metadata"
    # 查詢 Rag 表中 rag_id 符合且 deleted=False 的資料
    rag_rows = supabase.table("Rag").select(select_cols).eq("rag_id", rag_id).eq("deleted", False).execute()
    # 若查無資料，拋出 404
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    # 取第一筆資料
    row = rag_rows.data[0]
    # 取得 rag_metadata（可能為 dict 或 None）
    meta = row.get("rag_metadata")
    # 從 rag_metadata.outputs 取得 outputs 陣列；stem 來自 rag_tab_id（舊）/ unit_name / rag_name（舊）/ filename
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    # 若 outputs 為空，表示尚未執行 build-rag-zip，拋出 400
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")
    # repack stem：新資料為 unit_name + filename；更舊可能為 rag_tab_id 或 rag_name
    first = outputs[0] if isinstance(outputs[0], dict) else {}
    stem = (first.get("rag_tab_id") or first.get("unit_name") or first.get("rag_name") or "").strip()
    if not stem and first.get("filename"):
        stem = Path(str(first["filename"])).stem.strip()
    if not stem:
        raise HTTPException(
            status_code=400,
            detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少可辨識的 repack stem（unit_name 或 filename）",
        )
    # RAG ZIP 的 tab_id 為 stem 加 _rag 後綴
    rag_zip_tab_id = f"{stem}_rag"
    # 依據 include_row 回傳不同結構
    return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)
