"""
RAG 相關共用邏輯模組。
提供由 rag_id 查詢 Rag 表並取得 stem、rag_zip_tab_id 的函數。
"""

# 引入 FastAPI 的 HTTPException，用於拋出 404、400 等錯誤
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
    # 從 rag_metadata.outputs 取得 outputs 陣列，若無則為空列表
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    # 若 outputs 為空，表示尚未執行 build-rag-zip，拋出 400
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")
    # 從 outputs 第一筆取得 rag_tab_id 作為 stem，並去除前後空白
    stem = (outputs[0].get("rag_tab_id") or "").strip() if isinstance(outputs[0], dict) else ""
    # 若 stem 為空，拋出 400
    if not stem:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少 rag_tab_id")
    # RAG ZIP 的 tab_id 為 stem 加 _rag 後綴
    rag_zip_tab_id = f"{stem}_rag"
    # 依據 include_row 回傳不同結構
    return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)
