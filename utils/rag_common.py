"""RAG 相關共用邏輯：由 rag_id 查 Rag 表取得 stem、rag_zip_tab_id。"""

from fastapi import HTTPException


def get_rag_stem_from_rag_id(supabase, rag_id: int, include_row: bool = False):
    """
    由 rag_id 查 Rag 表，回傳 stem 與 rag_zip_tab_id。
    include_row=True 時回傳 (row, stem, rag_zip_tab_id)；否則回傳 (stem, rag_zip_tab_id)。
    """
    select_cols = "rag_tab_id, system_prompt_instruction, person_id, rag_id, rag_metadata" if include_row else "rag_metadata"
    rag_rows = supabase.table("Rag").select(select_cols).eq("rag_id", rag_id).eq("deleted", False).execute()
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    row = rag_rows.data[0]
    meta = row.get("rag_metadata")
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")
    stem = (outputs[0].get("rag_tab_id") or "").strip() if isinstance(outputs[0], dict) else ""
    if not stem:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少 rag_tab_id")
    rag_zip_tab_id = f"{stem}_rag"
    return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)
