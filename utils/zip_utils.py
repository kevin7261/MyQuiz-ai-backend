"""ZIP 檔案相關工具：編碼修正、第二層資料夾解析等。"""


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
