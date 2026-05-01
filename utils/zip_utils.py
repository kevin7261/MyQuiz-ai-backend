"""
ZIP 檔案相關工具。

提供編碼修正、第二層資料夾解析、依資料夾重新打包等功能。
"""

import io
import zipfile
from pathlib import Path


def fix_encoding(filename: str) -> str:
    """
    修正 ZIP 檔名亂碼。

    ZIP 規格早期以 cp437 儲存非 ASCII 檔名；嘗試以 cp437→utf-8 還原，
    失敗則再試 cp437→big5（繁體中文環境常見）。
    """
    try:
        return filename.encode("cp437").decode("utf-8")
    except Exception:
        try:
            return filename.encode("cp437").decode("big5")
        except Exception:
            return filename


def get_second_level_folders_from_zip_file(zip_file) -> list[str]:
    """
    從 ZIP 檔案物件讀取，回傳第二層資料夾名稱清單（排序、去重）。

    自動過濾 __MACOSX、.DS_Store，並對每個路徑套用 fix_encoding。
    """
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
    """
    將 (raw_name, decoded_name) 加入 folder_map[seg]，以 raw_name 去重。

    過濾無效或系統產生的路徑段（空字串、.、..、._*、__MACOSX）。
    """
    if not seg or seg in (".", "..") or seg.startswith("._") or seg == "__MACOSX":
        return
    lst = folder_map.setdefault(seg, [])
    if any(r == raw_name for r, _ in lst):
        return
    lst.append((raw_name, decoded_name))


def build_folder_map(zip_file) -> dict[str, list[tuple[str, str]]]:
    """
    建立「路徑上任意層資料夾名稱 → 該層之下所有檔案 (raw_name, decoded_name)」的對照表。

    對每個檔案，其路徑上每一層目錄名稱都會被註冊為 key，
    因此同一個資料夾可能由不同深度的路徑命中（向下相容舊版「6 位數日期資料夾」格式）。
    僅列檔案，不列以 / 結尾的目錄條目。
    """
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
    """
    自 repack 產出之 ZIP 檔名取得 stem（不含 .zip）。
    不可使用 Path(filename).stem：若檔名為 ``A/tB.zip``，Path 會誤取最後一段 ``tB``。
    """
    s = (filename or "").strip()
    if s.lower().endswith(".zip"):
        return s[:-4].strip()
    return Path(s).stem.strip()


def folder_combination_stem_from_targets(targets: list[str]) -> str:
    """
    多資料夾組合鍵（寫入 Rag_Unit.folder_combination、與 repack ZIP 檔名 stem 一致）：
    ``folder1/tfolder2/tfolder3``（第一個資料夾名後以 ``/t`` 連接其餘資料夾名）。
    單一資料夾時即該名稱本身。
    """
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
    """
    依 tasks 字串從 source ZIP 抽出指定資料夾，重新壓成多個 ZIP。

    格式：逗號分隔多個輸出 ZIP；加號表示同一 ZIP 內包含多個資料夾。
    範例："220222+220301"、"10_ERGMs"、"社會網絡分析,10_ERGMs"

    輸出 ZIP 檔名 stem 為 ``folder1/tfolder2/...``（多資料夾以 ``/t`` 連接），不再使用底線 ``_``。

    回傳 [(zip_bytes, filename), ...]；
    每個 ZIP 內的路徑只保留目標資料夾名稱該段之後（如 10_ERGMs/10_ERGMs.pdf）。
    """
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    if not tasks:
        return []

    result: list[tuple[bytes, str]] = []

    with zipfile.ZipFile(source_zip_path, "r") as z_source:
        for task in tasks:
            # 同一 ZIP 內可包含多個資料夾（以加號分隔）
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
                            # 只保留 target 資料夾名稱該段之後的相對路徑
                            clean_arcname = "/".join(parts[idx:])
                            z_out.writestr(clean_arcname, z_source.read(raw_path))
                        except (ValueError, KeyError):
                            continue
            result.append((buf.getvalue(), zip_filename))

    return result
