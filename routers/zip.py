"""ZIP 相關 API 路由。"""

import zipfile
from fastapi import APIRouter, HTTPException, UploadFile, File

from utils.zip_utils import get_second_level_folders_from_zip_file

router = APIRouter(prefix="/zip", tags=["zip"])


@router.post("/second-folders")
async def get_zip_second_folders(file: UploadFile = File(...)):
    """
    上傳 ZIP 檔案（由 API 傳入），回傳該 ZIP 內「第二層」資料夾名稱清單。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    try:
        with zipfile.ZipFile(file.file, "r") as zip_ref:
            folders = get_second_level_folders_from_zip_file(zip_ref)
        return {"filename": file.filename, "second_folders": folders}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取 ZIP 檔案")
