"""
ZIP 檔案相關工具模組。
提供編碼修正、第二層資料夾解析、依資料夾重新打包等功能。
"""

# 引入 io 模組的 BytesIO 等，用於記憶體中處理 ZIP
import io
# 引入 zipfile 模組，用於讀寫 ZIP 檔案
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path


def fix_encoding(filename: str) -> str:
    """
    修正 ZIP 檔名亂碼。
    嘗試將 cp437（ZIP 常見編碼）還原為 utf-8，失敗則試 big5。
    """
    try:
        # 先以 cp437 編碼再以 utf-8 解碼
        return filename.encode("cp437").decode("utf-8")
    except Exception:  # utf-8 失敗則試 big5
        try:
            # 失敗則試 big5（繁體中文常用）
            return filename.encode("cp437").decode("big5")
        except Exception:  # big5 也失敗
            # 仍失敗則回傳原檔名
            return filename


def get_second_level_folders_from_zip_file(zip_file) -> list[str]:
    """
    從 ZIP 檔案物件讀取，回傳第二層資料夾名稱清單。
    過濾 __MACOSX、.DS_Store，並修正編碼。
    """
    # 用 set 儲存不重複的第二層資料夾名稱
    second_folders = set()
    # 遍歷 ZIP 內所有檔名
    for name in zip_file.namelist():
        # 修正編碼
        real_name = fix_encoding(name)
        # 跳過 macOS 隱藏檔
        if "__MACOSX" in real_name or ".DS_Store" in real_name:
            continue
        # 以 / 分割路徑，取至少兩層（如 a/b/c -> ["a","b","c"]）
        parts = real_name.strip("/").split("/")
        if len(parts) >= 2:
            # 加入第二層（parts[1]）
            second_folders.add(parts[1])
    # 排序後以 list 回傳
    return sorted(list(second_folders))


def build_folder_map(zip_file) -> dict[str, list[tuple[str, str]]]:
    """
    建立「6 位數資料夾名稱 → 該資料夾內檔案 (raw_name, decoded_name)」的對照表。
    過濾 __MACOSX、.DS_Store，僅列檔案（不列目錄結尾）。
    """
    # 初始化對照表
    folder_map: dict[str, list[tuple[str, str]]] = {}
    # 遍歷 ZIP 內所有檔名
    for raw_name in zip_file.namelist():
        # 跳過目錄（以 / 結尾）
        if raw_name.endswith("/"):
            continue
        # 修正編碼取得 decoded_name
        decoded_name = fix_encoding(raw_name)
        # 跳過 macOS 隱藏檔
        if "__MACOSX" in decoded_name or ".DS_Store" in decoded_name:
            continue
        # 分割路徑
        parts = decoded_name.split("/")
        # 找出路徑中第一個 6 位數字（如 220222）
        for part in parts:
            if part.isdigit() and len(part) == 6:
                target = part
                # 若該資料夾尚未在 map 中，建立空列表
                if target not in folder_map:
                    folder_map[target] = []
                # 加入 (raw_name, decoded_name)
                folder_map[target].append((raw_name, decoded_name))
                break
    return folder_map


def repack_tasks_to_zips(
    source_zip_path: Path,
    folder_map: dict[str, list[tuple[str, str]]],
    tasks_str: str,
) -> list[tuple[bytes, str]]:
    """
    依 tasks 字串從 source zip 抽出指定資料夾，重新壓成多個 ZIP。
    格式：逗號分隔多個輸出檔，加號為同一檔內多個資料夾。例："220222+220301" 或 "220222,220301+220302"
    回傳 [(zip_bytes, filename), ...]，路徑只保留目標資料夾之後（如 220222/Class/file.pdf）。
    """
    # 以逗號分割 tasks，去除空白
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    if not tasks:
        return []

    result: list[tuple[bytes, str]] = []

    with zipfile.ZipFile(source_zip_path, "r") as z_source:
        # 逐一處理每個 task（每個 task 對應一個輸出 ZIP）
        for task in tasks:
            # 以加號分割，取得同一 ZIP 內要包含的資料夾
            targets = [t.strip() for t in task.split("+") if t.strip()]
            if not targets:
                continue
            # 檔名由資料夾名以底線連接，如 220222_220301.zip
            zip_filename = "_".join(targets) + ".zip"
            # 建立記憶體中的 ZIP
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z_out:
                for target in targets:
                    # 從 folder_map 取得該資料夾內所有檔案的 (raw_path, decoded_path)
                    for raw_path, decoded_path in folder_map.get(target, []):
                        parts = decoded_path.split("/")
                        try:
                            # 找到 target 在路徑中的索引
                            idx = parts.index(target)
                            # 只保留 target 之後的相對路徑（如 220222/Class/file.pdf）
                            clean_arcname = "/".join(parts[idx:])
                            # 從來源 ZIP 讀取內容並寫入輸出 ZIP
                            z_out.writestr(clean_arcname, z_source.read(raw_path))
                        except (ValueError, KeyError):
                            continue
            # 將 ZIP bytes 加入結果
            result.append((buf.getvalue(), zip_filename))

    return result
