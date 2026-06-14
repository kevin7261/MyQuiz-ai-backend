"""
Bank 專屬 ZIP 工具（自 utils.zip_utils 複製，與 rag 無關）。

提供編碼修正、第二層資料夾解析、依資料夾重新打包等功能。
"""

import io
import zipfile
from pathlib import Path


def fix_encoding(filename: str) -> str:
    """修正 ZIP 檔名亂碼（cp437→utf-8，失敗再依序試 big5、gbk）。"""
    try:
        return filename.encode("cp437").decode("utf-8")
    except Exception:
        for enc in ("big5", "gbk"):
            try:
                return filename.encode("cp437").decode(enc)
            except Exception:
                continue
        return filename


def get_second_level_folders_from_zip_file(zip_file) -> list[str]:
    """回傳第二層資料夾名稱清單（排序、去重）；過濾 __MACOSX、.DS_Store。"""
    second_folders: set[str] = set()
    for name in zip_file.namelist():
        real_name = fix_encoding(name)
        if "__MACOSX" in real_name or ".DS_Store" in real_name:
            continue
        parts = real_name.strip("/").split("/")
        if len(parts) >= 2:
            second_folders.add(parts[1])
    return sorted(list(second_folders))


def _folder_map_append(
    folder_map: dict[str, list[tuple[str, str]]],
    seg: str,
    raw_name: str,
    decoded_name: str,
) -> None:
    if not seg or seg in (".", "..") or seg.startswith("._") or seg == "__MACOSX":
        return
    lst = folder_map.setdefault(seg, [])
    if any(r == raw_name for r, _ in lst):
        return
    lst.append((raw_name, decoded_name))


def build_folder_map(zip_file) -> dict[str, list[tuple[str, str]]]:
    """建立「路徑上任意層資料夾名稱 → 該層之下所有檔案 (raw_name, decoded_name)」對照表。"""
    folder_map: dict[str, list[tuple[str, str]]] = {}
    for raw_name in zip_file.namelist():
        if raw_name.endswith("/"):
            continue
        decoded_name = fix_encoding(raw_name)
        if "__MACOSX" in decoded_name or ".DS_Store" in decoded_name:
            continue
        parts = decoded_name.split("/")
        if len(parts) < 2:
            continue
        for i in range(len(parts) - 1):
            seg = parts[i]
            if seg == ".DS_Store":
                continue
            _folder_map_append(folder_map, seg, raw_name, decoded_name)
    return folder_map


def repack_zip_stem_from_filename(filename: str) -> str:
    """自 repack 產出之 ZIP 檔名取得 stem（不含 .zip）。"""
    s = (filename or "").strip()
    if s.lower().endswith(".zip"):
        return s[:-4].strip()
    return Path(s).stem.strip()


def folder_combination_stem_from_targets(targets: list[str]) -> str:
    """多資料夾組合鍵（寫入 Bank_Unit.folder_combination、與 repack ZIP 檔名 stem 一致）：``folder1/tfolder2``。"""
    cleaned = [t.strip() for t in targets if (t or "").strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned[0] + "".join(f"/t{x}" for x in cleaned[1:])


def repack_tasks_to_zips(
    source_zip_path: Path,
    folder_map: dict[str, list[tuple[str, str]]],
    tasks_str: str,
) -> list[tuple[bytes, str]]:
    """依 tasks 字串從 source ZIP 抽出指定資料夾，重新壓成多個 ZIP。回傳 [(zip_bytes, filename), ...]。"""
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    if not tasks:
        return []

    result: list[tuple[bytes, str]] = []

    with zipfile.ZipFile(source_zip_path, "r") as z_source:
        for task in tasks:
            targets = [t.strip() for t in task.split("+") if t.strip()]
            if not targets:
                continue
            zip_filename = folder_combination_stem_from_targets(targets) + ".zip"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z_out:
                for target in targets:
                    for raw_path, decoded_path in folder_map.get(target, []):
                        parts = decoded_path.split("/")
                        try:
                            idx = parts.index(target)
                            clean_arcname = "/".join(parts[idx:])
                            z_out.writestr(clean_arcname, z_source.read(raw_path))
                        except (ValueError, KeyError):
                            continue
            result.append((buf.getvalue(), zip_filename))

    return result
