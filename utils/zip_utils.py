"""ZIP 檔案相關工具：編碼修正、第二層資料夾解析、依資料夾重新打包等。"""

import io
import zipfile
from pathlib import Path


def fix_encoding(filename: str) -> str:
    """修正亂碼：嘗試將 cp437 編碼還原為 utf-8 或 big5"""
    try:
        return filename.encode("cp437").decode("utf-8")
    except Exception:
        try:
            return filename.encode("cp437").decode("big5")
        except Exception:
            return filename


def get_second_level_folders_from_zip_file(zip_file) -> list[str]:
    """
    從 ZIP 檔案物件（API 傳入）讀取，回傳第二層資料夾名稱清單（不重複、已排序）。
    過濾 __MACOSX、.DS_Store，並修正編碼。
    """
    second_folders = set()
    for name in zip_file.namelist():
        real_name = fix_encoding(name)
        if "__MACOSX" in real_name or ".DS_Store" in real_name:
            continue
        parts = real_name.strip("/").split("/")
        if len(parts) >= 2:
            second_folders.add(parts[1])
    return sorted(list(second_folders))


def build_folder_map(zip_file) -> dict[str, list[tuple[str, str]]]:
    """
    建立「6 位數資料夾名稱 → 該資料夾內檔案 (raw_name, decoded_name)」的對照。
    過濾 __MACOSX、.DS_Store，僅列檔案（不列目錄結尾）。
    """
    folder_map: dict[str, list[tuple[str, str]]] = {}
    for raw_name in zip_file.namelist():
        if raw_name.endswith("/"):
            continue
        decoded_name = fix_encoding(raw_name)
        if "__MACOSX" in decoded_name or ".DS_Store" in decoded_name:
            continue
        parts = decoded_name.split("/")
        for part in parts:
            if part.isdigit() and len(part) == 6:
                target = part
                if target not in folder_map:
                    folder_map[target] = []
                folder_map[target].append((raw_name, decoded_name))
                break
    return folder_map


def repack_tasks_to_zips(
    source_zip_path: Path,
    folder_map: dict[str, list[tuple[str, str]]],
    tasks_str: str,
) -> list[tuple[bytes, str]]:
    """
    依 tasks 字串（例："220222+220301" 或 "220222,220301+220302"）從 source zip 抽出指定資料夾，
    重新壓成多個 ZIP。逗號分隔多個輸出檔，加號為同一檔內多個資料夾。
    回傳 [(zip_bytes, filename), ...]，路徑只保留目標資料夾之後（如 220222/Class/file.pdf）。
    """
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    if not tasks:
        return []

    result: list[tuple[bytes, str]] = []

    with zipfile.ZipFile(source_zip_path, "r") as z_source:
        for task in tasks:
            targets = [t.strip() for t in task.split("+") if t.strip()]
            if not targets:
                continue
            zip_filename = "_".join(targets) + ".zip"
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
