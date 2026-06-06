"""
依 Rag 表與 Rag_Unit 表解析 repack stem、rag_zip_page_id（供取得 RAG ZIP 路徑等）。
優先從 Rag_Unit 讀取（新版 schema）；Rag_Unit 無資料時回退至 rag_metadata.outputs（向下相容舊版）。
"""

from pathlib import Path

from fastapi import HTTPException
from postgrest.exceptions import APIError

from utils.zip_utils import repack_zip_stem_from_filename


def transcript_from_row(row: dict | None) -> str:
    """Rag / Rag_Unit 逐字稿欄位（新欄 transcript；舊欄 transcription 向下相容）。"""
    if not row:
        return ""
    return (row.get("transcript") or row.get("transcription") or "").strip()


def instruction_from_rag_row(row: dict | None) -> str:
    """Rag 表層說明文：**Rag.transcript**（與出題時注入 LLM 的內容一致）。"""
    return transcript_from_row(row)


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
    stem = (entry.get("folder_combination") or "").strip()
    if stem:
        return stem
    stem = (
        entry.get("rag_page_id")
        or entry.get("unit_name")
        or entry.get("tab_name")
        or entry.get("rag_name")
        or ""
    ).strip()
    if not stem and entry.get("filename"):
        stem = repack_zip_stem_from_filename(str(entry["filename"]))
    return stem


def _output_unit_candidates(entry: dict) -> set[str]:
    """可供前端傳入比對的 unit 識別字串。"""
    out: set[str] = set()
    if not isinstance(entry, dict):
        return out
    for k in ("folder_combination", "unit_name", "rag_page_id", "tab_name", "rag_name"):
        v = (entry.get(k) or "").strip()
        if v:
            out.add(v)
    fn = entry.get("filename")
    if fn:
        st = repack_zip_stem_from_filename(str(fn))
        if st:
            out.add(st)
    return out


def _stem_from_rag_file_name(rag_file_name: str, unit_name: str) -> tuple[str, str]:
    """
    從 rag_file_name（例如 abc123_rag.zip）解析 stem 與 rag_zip_page_id。
    rag_file_name 對應 Rag_Unit.rag_file_name，格式為 {stem}_rag.zip。
    回傳 (stem, rag_zip_page_id)；無法解析時以 unit_name 組合。
    """
    if rag_file_name:
        page_id = Path(rag_file_name).stem  # e.g., "abc123_rag"
        if page_id.endswith("_rag"):
            stem = page_id[:-4]
        else:
            stem = unit_name
            page_id = f"{unit_name}_rag"
        return stem, page_id
    stem = unit_name
    return stem, f"{stem}_rag"


def get_rag_stem_from_rag_id(
    supabase,
    rag_id: int,
    include_row: bool = False,
    unit_name: str | None = None,
    rag_unit_id: int | None = None,
):
    """
    由 rag_id 查詢 Rag 表，再從 Rag_Unit 表（優先）或 rag_metadata.outputs（向下相容）取得
    repack stem 與 rag_zip_page_id（通常為 {stem}_rag）。

    include_row=True 時回傳 (row, stem, rag_zip_page_id)；否則回傳 (stem, rag_zip_page_id)。
    rag_unit_id 若 >0，優先選取該主鍵之列（須隸屬此 Rag.rag_page_id）。
    否則 unit_name 若指定（非空白），則選取該名稱的單元；皆未指定則使用第一筆。
    """
    # 勿在主要 SELECT 含 rag_metadata：部分環境尚未 migration 該欄，會導致整筆查詢 42703。
    # 部分環境 Rag 表尚無 transcript 欄位（42703）時改選不含該欄。
    select_cols = (
        "rag_page_id, transcript, person_id, rag_id"
        if include_row
        else "rag_page_id"
    )
    select_cols_no_transcript = (
        "rag_page_id, person_id, rag_id" if include_row else "rag_page_id"
    )
    select_cols_legacy_transcription = (
        "rag_page_id, transcription, person_id, rag_id" if include_row else "rag_page_id"
    )
    try:
        rag_rows = (
            supabase.table("Rag")
            .select(select_cols)
            .eq("rag_id", rag_id)
            .eq("deleted", False)
            .execute()
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "transcript" in msg and include_row:
            try:
                rag_rows = (
                    supabase.table("Rag")
                    .select(select_cols_legacy_transcription)
                    .eq("rag_id", rag_id)
                    .eq("deleted", False)
                    .execute()
                )
            except APIError as e2:
                msg2 = (e2.message or "").lower()
                if e2.code == "42703" and "transcription" in msg2:
                    rag_rows = (
                        supabase.table("Rag")
                        .select(select_cols_no_transcript)
                        .eq("rag_id", rag_id)
                        .eq("deleted", False)
                        .execute()
                    )
                else:
                    raise
        else:
            raise
    if not rag_rows.data or len(rag_rows.data) == 0:
        raise HTTPException(status_code=404, detail=f"找不到 rag_id={rag_id} 的 Rag 資料")
    row = rag_rows.data[0]
    rag_page_id = (row.get("rag_page_id") or "").strip()

    try:
        unit_rows = (
            supabase.table("Rag_Unit")
            .select("rag_unit_id, unit_name, folder_combination, rag_file_name, repack_file_name")
            .eq("rag_page_id", rag_page_id)
            .eq("deleted", False)
            .order("created_at", desc=False)
            .execute()
        )
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            unit_rows = (
                supabase.table("Rag_Unit")
                .select("rag_unit_id, unit_name, rag_file_name, repack_file_name")
                .eq("rag_page_id", rag_page_id)
                .eq("deleted", False)
                .order("created_at", desc=False)
                .execute()
            )
        else:
            raise
    units = unit_rows.data or []

    if units:
        available = [
            (u.get("unit_name") or u.get("folder_combination") or "?").strip() or "?"
            for u in units
        ]
        wanted = (unit_name or "").strip()
        try:
            ruid_wanted = int(rag_unit_id) if rag_unit_id is not None else 0
        except (TypeError, ValueError):
            ruid_wanted = 0
        selected: dict | None = None
        if ruid_wanted > 0:
            for u in units:
                try:
                    if int(u.get("rag_unit_id") or 0) == ruid_wanted:
                        selected = u
                        break
                except (TypeError, ValueError):
                    continue
            if selected is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"找不到 rag_unit_id={ruid_wanted} 的 Rag_Unit（rag_page_id={rag_page_id!r}）；"
                        f"可選 unit_name：{available}"
                    ),
                )
        elif not wanted:
            selected = units[0]
        else:
            for u in units:
                un = (u.get("unit_name") or "").strip()
                fc = (u.get("folder_combination") or "").strip()
                if un == wanted or fc == wanted:
                    selected = u
                    break
            if selected is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"找不到 unit_name／folder_combination={wanted!r} 的 Rag_Unit；可選：{available}",
                )
        path_key = (selected.get("folder_combination") or selected.get("unit_name") or "").strip()
        stem, rag_zip_page_id = _stem_from_rag_file_name(
            selected.get("rag_file_name", ""),
            path_key,
        )
        if not stem:
            raise HTTPException(
                status_code=400,
                detail=f"Rag_Unit（rag_page_id={rag_page_id}）缺少可辨識的 stem",
            )
        return (row, stem, rag_zip_page_id) if include_row else (stem, rag_zip_page_id)

    meta = _fetch_rag_metadata_if_present(supabase, rag_id)
    if include_row and meta is not None:
        row = {**row, "rag_metadata": meta}
    outputs = (meta.get("outputs", []) if isinstance(meta, dict) else []) or []
    if not outputs:
        raise HTTPException(
            status_code=400,
            detail=f"該筆 Rag（rag_id={rag_id}）尚無 Rag_Unit 資料，請先執行 POST /v1/rag/pages/{rag_page_id}/build-zip",
        )

    wanted = (unit_name or "").strip()
    stem = ""
    matched_output: dict | None = None
    if not wanted:
        matched_output = outputs[0] if isinstance(outputs[0], dict) else None
        stem = _stem_from_output_entry(matched_output or {})
    else:
        for o in outputs:
            if not isinstance(o, dict):
                continue
            if wanted in _output_unit_candidates(o):
                matched_output = o
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
    rag_zip_page_id = ""
    src = matched_output if matched_output is not None else (
        outputs[0] if isinstance(outputs[0], dict) else None
    )
    if isinstance(src, dict):
        rf = (src.get("rag_filename") or "").strip()
        if rf.lower().endswith(".zip"):
            rid = rf[:-4].strip()
            if rid.endswith("_rag") and "/" not in rid and "\\" not in rid:
                rag_zip_page_id = rid
    if not rag_zip_page_id:
        if "/" in stem or "\\" in stem:
            raise HTTPException(
                status_code=400,
                detail=(
                    "此 Rag 僅有舊版 rag_metadata.outputs，且 folder_combination 含路徑字元，"
                    "無法自 stem 組出 rag ZIP page_id；請改用含 Rag_Unit 的資料或傳 rag_unit_id"
                ),
            )
        rag_zip_page_id = f"{stem}_rag"
    return (row, stem, rag_zip_page_id) if include_row else (stem, rag_zip_page_id)
