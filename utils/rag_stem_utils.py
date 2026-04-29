"""
依 Rag 表與 Rag_Unit 表解析 repack stem、rag_zip_tab_id（供取得 RAG ZIP 路徑等）。
優先從 Rag_Unit 讀取（新版 schema）；Rag_Unit 無資料時回退至 rag_metadata.outputs（向下相容舊版）。
"""

from pathlib import Path

from fastapi import HTTPException
from postgrest.exceptions import APIError


def instruction_from_rag_row(row: dict | None) -> str:
    """Rag 表層級補充文字：**Rag.transcription**（與出題時注入 LLM 的內容一致）。"""
    if not row:
        return ""
    return (row.get("transcription") or "").strip()


def _fetch_rag_metadata_if_present(supabase, rag_id: int):
    """
    僅在需要 rag_metadata.outputs 回退時查詢；若資料庫尚無 rag_metadata 欄位則回傳 None。
    """
    try:
        resp = (
            supabase.table("Rag")
            .select("rag_metadata")
            .eq("rag_id", rag_id)
            .eq("deleted", False)
            .execute()
        )
        if resp.data:
            return resp.data[0].get("rag_metadata")
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "rag_metadata" in msg:
            return None
        raise
    return None


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
    """可供前端傳入比對的 unit 識別字串。"""
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


def _stem_from_rag_file_name(rag_file_name: str, unit_name: str) -> tuple[str, str]:
    """
    從 rag_file_name（例如 abc123_rag.zip）解析 stem 與 rag_zip_tab_id。
    rag_file_name 對應 Rag_Unit.rag_file_name，格式為 {stem}_rag.zip。
    回傳 (stem, rag_zip_tab_id)；無法解析時以 unit_name 組合。
    """
    if rag_file_name:
        tab_id = Path(rag_file_name).stem  # e.g., "abc123_rag"
        if tab_id.endswith("_rag"):
            stem = tab_id[:-4]
        else:
            stem = unit_name
            tab_id = f"{unit_name}_rag"
        return stem, tab_id
    stem = unit_name
    return stem, f"{stem}_rag"


def get_rag_stem_from_rag_id(
    supabase,
    rag_id: int,
    include_row: bool = False,
    unit_name: str | None = None,
):
    """
    由 rag_id 查詢 Rag 表，再從 Rag_Unit 表（優先）或 rag_metadata.outputs（向下相容）取得
    repack stem 與 rag_zip_tab_id（通常為 {stem}_rag）。

    include_row=True 時回傳 (row, stem, rag_zip_tab_id)；否則回傳 (stem, rag_zip_tab_id)。
    unit_name 若指定（非空白），則選取該名稱的單元；未指定則使用第一筆。
    """
    # 勿在主要 SELECT 含 rag_metadata：部分環境尚未 migration 該欄，會導致整筆查詢 42703。
    select_cols = (
        "rag_tab_id, transcription, person_id, rag_id"
        if include_row
        else "rag_tab_id"
    )
    rag_rows = (
        supabase.table("Rag")
        .select(select_cols)
        .eq("rag_id", rag_id)
        .eq("deleted", False)
        .execute()
    )
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    row = rag_rows.data[0]
    rag_tab_id = (row.get("rag_tab_id") or "").strip()

    unit_rows = (
        supabase.table("Rag_Unit")
        .select("rag_unit_id, unit_name, rag_file_name, repack_file_name")
        .eq("rag_tab_id", rag_tab_id)
        .eq("deleted", False)
        .order("created_at", desc=False)
        .execute()
    )
    units = unit_rows.data or []

    if units:
        wanted = (unit_name or "").strip()
        selected: dict | None = None
        if not wanted:
            selected = units[0]
        else:
            for u in units:
                if (u.get("unit_name") or "").strip() == wanted:
                    selected = u
                    break
            if selected is None:
                available = [(u.get("unit_name") or "?") for u in units]
                raise HTTPException(
                    status_code=400,
                    detail=f"找不到 unit_name={wanted!r} 的 Rag_Unit；可選：{available}",
                )
        stem, rag_zip_tab_id = _stem_from_rag_file_name(
            selected.get("rag_file_name", ""),
            selected.get("unit_name", ""),
        )
        if not stem:
            raise HTTPException(
                status_code=400,
                detail=f"Rag_Unit（rag_tab_id={rag_tab_id}）缺少可辨識的 stem",
            )
        return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)

    meta = _fetch_rag_metadata_if_present(supabase, rag_id)
    if include_row and meta is not None:
        row = {**row, "rag_metadata": meta}
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        raise HTTPException(
            status_code=400,
            detail=f"該筆 Rag（rag_id={rag_id}）尚無 Rag_Unit 資料，請先執行 POST /rag/tab/build-rag-zip",
        )

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
            detail=f"該筆 Rag（rag_id={rag_id}）的 outputs 第一筆缺少可辨識的 repack stem",
        )
    rag_zip_tab_id = f"{stem}_rag"
    return (row, stem, rag_zip_tab_id) if include_row else (stem, rag_zip_tab_id)
