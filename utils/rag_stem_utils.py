"""
依 Rag 表與 rag_metadata.outputs 解析 repack stem、rag_zip_tab_id（供取得 RAG ZIP 路徑等）。
"""

# 引入 FastAPI 的 HTTPException，用於拋出 404、400 等錯誤
from pathlib import Path

from fastapi import HTTPException


def _stem_from_output_entry(entry: dict) -> str:
    """由單一 outputs[] 項目取得 repack stem（與 RAG ZIP 的 {stem}_rag 一致）。"""
    if not isinstance(entry, dict):
        return ""
    stem = (
        entry.get("rag_tab_id")
        or entry.get("unit_name")
        or entry.get("tab_name")
        or entry.get("rag_name")
        or ""
    ).strip()
    if not stem and entry.get("filename"):
        stem = Path(str(entry["filename"])).stem.strip()
    return stem


def _output_unit_candidates(entry: dict) -> set[str]:
    """可供前端傳入比對的 unit 識別字串（unit_name、舊欄位、檔名不含副檔名）。"""
    out: set[str] = set()
    if not isinstance(entry, dict):
        return out
    for k in ("unit_name", "rag_tab_id", "tab_name", "rag_name"):
        v = (entry.get(k) or "").strip()
        if v:
            out.add(v)
    fn = entry.get("filename")
    if fn:
        st = Path(str(fn)).stem.strip()
        if st:
            out.add(st)
    return out


def get_rag_stem_from_rag_id(
    supabase,
    rag_id: int,
    include_row: bool = False,
    unit_name: str | None = None,
):
    """
    由 rag_id 查詢 Rag 表，回傳 stem（RAG ZIP 的 tab_id）與 rag_zip_tab_id（通常為 {stem}_rag）。
    include_row=True 時回傳 (row, stem, rag_zip_tab_id)；否則回傳 (stem, rag_zip_tab_id)。
    unit_name 若指定（非空白），則自 rag_metadata.outputs 選取該上傳單元（與 build-rag-zip 的 outputs[].unit_name 等一致）；未指定則使用第一筆輸出。
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
    # 從 rag_metadata.outputs 取得 outputs 陣列；stem 來自 rag_tab_id（舊）/ unit_name / tab_name / rag_name（舊）/ filename
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    # 若 outputs 為空，表示尚未執行 build-rag-zip，拋出 400
    if not outputs:
        raise HTTPException(status_code=400, detail=f"該筆 Rag（rag_id={rag_id}）的 rag_metadata.outputs 為空，請先執行 build-rag-zip")

    wanted = (unit_name or "").strip()
    stem = ""
    if not wanted:
        first = outputs[0] if isinstance(outputs[0], dict) else {}
        stem = _stem_from_output_entry(first)
    else:
        for o in outputs:
            if not isinstance(o, dict):
                continue
            if wanted in _output_unit_candidates(o):
                stem = _stem_from_output_entry(o) or wanted
                break
        if not stem:
            available: list[str] = []
            for o in outputs:
                if isinstance(o, dict):
                    u = (o.get("unit_name") or "").strip()
                    s = _stem_from_output_entry(o)
                    available.append(u or s or "?")
            raise HTTPException(
                status_code=400,
                detail=f"找不到 unit_name={wanted!r} 的輸出單元；可選：{available}",
            )

    if not stem:
        raise HTTPException(
            status_code=400,
            detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少可辨識的 repack stem（unit_name 或 filename）",
        )
    # RAG ZIP 的 tab_id 為 stem 加 _rag 後綴
    rag_zip_tab_id = f"{stem}_rag"
    # 依據 include_row 回傳不同結構
    return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)
