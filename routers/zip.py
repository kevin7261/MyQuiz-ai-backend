"""ZIP 相關 API 路由。"""

import io
import zipfile
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
)
from utils.storage import save_zip, get_zip_path

router = APIRouter(prefix="/zip", tags=["zip"])


class PackRequest(BaseModel):
    """指定先前上傳的 ZIP（file_id）與要打包的資料夾規則。"""
    file_id: str
    tasks: str  # 例："220222+220301" 或 "220222,220301+220302"（逗號=多個 ZIP，加號=同一 ZIP 多資料夾）


@router.post("/second-folders")
async def get_zip_second_folders(file: UploadFile = File(...)):
    """
    上傳 ZIP 檔案（由 API 傳入），會存到後端空間，回傳 file_id、第二層資料夾清單。
    其他 API 可用 utils.storage.get_zip_path(file_id) 取得檔案路徑後讀取。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            folders = get_second_level_folders_from_zip_file(zip_ref)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取 ZIP 檔案")

    file_id = save_zip(contents, file.filename)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "second_folders": folders,
    }


@router.post("/pack")
def pack_folders(body: PackRequest):
    """
    依先前上傳的 ZIP（file_id）與 tasks 字串，抽出指定 6 位數資料夾重新壓成 ZIP 並存到後端。
    tasks 格式：逗號分隔多個輸出檔，加號為同一檔內多個資料夾。
    例："220222+220301" → 一個 ZIP；"220222,220301+220302" → 兩個 ZIP。
    回傳新產生的 file_id、filename，可供下載或給其他 API 使用。
    """
    path = get_zip_path(body.file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該上傳的 ZIP，請先上傳或確認 file_id")

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.tasks)
    if not packed:
        raise HTTPException(status_code=400, detail="tasks 為空或格式錯誤，例：220222+220301")

    outputs = []
    for zip_bytes, filename in packed:
        file_id = save_zip(zip_bytes, filename)
        outputs.append({"file_id": file_id, "filename": filename})

    return {"source_file_id": body.file_id, "outputs": outputs}
