"""ZIP 相關 API 路由。"""

import io
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
)
from utils.storage import (
    save_zip,
    get_zip_path,
    get_zip_filename,
    FOLDER_UPLOAD,
    FOLDER_REPACK,
    FOLDER_RAG,
)
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/zip", tags=["zip"])

# Rag 表列出全部資料時選取欄位（用 * 回傳全部欄位）
RAG_SELECT_ALL = "*"


def _rag_default_row(
    file_id: str,
    *,
    person_id: str | None = None,
    file_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rag 表一筆新增用的預設欄位，create-rag 與 upload-zip 共用。"""
    row: dict[str, Any] = {
        "file_id": file_id,
        "file_metadata": file_metadata,
        "rag_list": "",
        "rag_metadata": None,
        "chunk_size": 0,
        "chunk_overlap": 0,
        "deleted": False,
    }
    if person_id is not None:
        row["person_id"] = person_id
    return row


def _rag_table_select(select_spec: str = "*") -> list[dict]:
    """查詢 Rag 表全部列（表名先試 "Rag"，失敗則試 "rag"）。回傳 resp.data 列表。"""
    supabase = get_supabase()
    try:
        resp = supabase.table("Rag").select(select_spec).execute()
        return resp.data or []
    except Exception as e:
        if "relation" in str(e).lower() or "does not exist" in str(e).lower():
            resp = get_supabase().table("rag").select(select_spec).execute()
            return resp.data or []
        raise


class ListRagResponse(BaseModel):
    """GET /zip/rag 回應：Rag 表全部資料。"""
    rags: list[dict]
    count: int


class PackRequest(BaseModel):
    """指定先前上傳的 ZIP（file_id）與要打包的資料夾規則。"""
    file_id: str
    tasks: str  # 例："220222+220301" 或 "220222,220301+220302"（逗號=多個 ZIP，加號=同一 ZIP 多資料夾）
    with_rag: bool = False  # 若 True，每個壓縮檔都會再做成 RAG（FAISS）ZIP，並回傳下載連結
    openai_api_key: str | None = None  # with_rag=True 時必填，用於 Embedding（不從環境變數讀取）
    chunk_size: int = 1000  # RAG 文件切分區塊大小
    chunk_overlap: int = 200  # RAG 切分區塊重疊字數


class GenerateQuestionRequest(BaseModel):
    """指定 RAG ZIP 的 file_id 與出題參數（由 API 傳入 openai_api_key）。"""
    file_id: str  # 使用者選擇的 RAG zip 的 file_id（pack 回傳的 rag_file_id）
    openai_api_key: str  # 用於 GPT-4o 出題，不從環境變數讀取
    qtype: str  # 題型
    level: str  # 難度


def _resolve_person_id(form_person_id: str | None, x_person_id: str | None) -> str | None:
    """優先 Form person_id，若無則 Header X-Person-Id。回傳非空字串或 None。"""
    for raw in (form_person_id, x_person_id):
        if raw is not None and raw.strip():
            return raw.strip()
    return None


@router.get("/rag", response_model=ListRagResponse)
def list_rag():
    """
    列出 Rag 表全部內容（與 GET /users 一樣回傳全部資料）。
    """
    try:
        data = _rag_table_select(RAG_SELECT_ALL)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-rag")
def create_rag(
    file_id: str = Form(..., description="Rag 紀錄的 file_id（由 API 傳入，可與上傳 ZIP 的 file_id 一致）"),
    person_id: str | None = Form(None, description="寫入 Rag 表的 person_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    傳入 file_id、person_id 建立或更新 Rag 表一筆資料（upsert）。
    person_id 可由 Form 或 Header X-Person-Id 傳入。
    若該 file_id 已存在則只更新 person_id；若不存在則新增一筆 Rag 紀錄。
    上傳 ZIP 為獨立流程，不要求先上傳再呼叫此 API。
    """
    fid = (file_id or "").strip()
    if not fid:
        raise HTTPException(status_code=400, detail="請傳入 file_id")
    resolved_person_id = _resolve_person_id(person_id, x_person_id)
    if not resolved_person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id（Form 或 Header X-Person-Id）")

    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .update({"person_id": resolved_person_id})
            .eq("file_id", fid)
            .execute()
        )
        if r.data and len(r.data) > 0:
            return {"file_id": fid, "person_id": resolved_person_id, "ok": True}
        # 無該 file_id：新增一筆 Rag 紀錄
        supabase.table("Rag").insert(
            _rag_default_row(fid, person_id=resolved_person_id)
        ).execute()
        return {"file_id": fid, "person_id": resolved_person_id, "ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    file_id: str = Form(..., description="由 API 傳入的 file_id，用於儲存與 file_metadata"),
    person_id: str | None = Form(None, description="僅用於儲存路徑，不寫入 Rag 表"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    上傳 ZIP 檔案（由 API 傳入），存到後端空間，並寫入 Rag 表該 file_id 的 file_metadata 欄位。
    回傳 file_id、第二層資料夾清單。person_id 僅用於儲存路徑（可選）。
    建立 RAG 一筆資料（寫入 person_id）請呼叫 POST /zip/create-rag。
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

    file_id = (file_id or "").strip()
    if not file_id:
        raise HTTPException(status_code=400, detail="請傳入 file_id")

    resolved_person_id = _resolve_person_id(person_id, x_person_id)
    try:
        file_id = save_zip(
            contents,
            file.filename,
            folder=FOLDER_UPLOAD,
            person_id=resolved_person_id,
            file_id=file_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = {
        "file_id": file_id,
        "filename": file.filename,
        "second_folders": folders,
    }

    # 僅寫入 Rag 表該 file_id 的 file_metadata（upsert：有則更新，無則新增一筆）
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .update({"file_metadata": response})
            .eq("file_id", file_id)
            .execute()
        )
        if not r.data or len(r.data) == 0:
            supabase.table("Rag").insert(
                _rag_default_row(file_id, file_metadata=response)
            ).execute()
    except Exception:
        pass

    return response


@router.get("/download/{file_id}")
def download_zip(file_id: str):
    """
    依 file_id 下載已儲存的 ZIP 檔。下載連結可供 pack 回傳的 rag_download_url 使用。
    """
    path = get_zip_path(file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該檔案")
    filename = get_zip_filename(file_id) or f"{file_id}.zip"
    return FileResponse(path, filename=filename, media_type="application/zip")


@router.post("/pack")
def pack_folders(request: Request, body: PackRequest):
    """
    依先前上傳的 ZIP（file_id）與 tasks 字串，抽出指定 6 位數資料夾重新壓成 ZIP 並存到後端。
    tasks 格式：逗號分隔多個輸出檔，加號為同一檔內多個資料夾。
    若 with_rag=True，每個壓縮檔會再做成 RAG（FAISS）ZIP，並回傳下載連結。
    """
    path = get_zip_path(body.file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該上傳的 ZIP，請先上傳或確認 file_id")

    # ZIP 永久保留，不再清空 repack / rag
    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.tasks)
    if not packed:
        raise HTTPException(status_code=400, detail="tasks 為空或格式錯誤，例：220222+220301")

    api_key = body.openai_api_key.strip() if (body.openai_api_key and body.openai_api_key.strip()) else None
    if body.with_rag and not api_key:
        raise HTTPException(status_code=400, detail="with_rag 為 true 時請傳入 openai_api_key")

    base_url = str(request.base_url).rstrip("/")

    outputs = []
    for zip_bytes, filename in packed:
        file_id = save_zip(zip_bytes, filename, folder=FOLDER_REPACK)
        item = {
            "file_id": file_id,
            "filename": filename,
            "download_url": f"{base_url}/zip/download/{file_id}",
        }
        if body.with_rag:
            try:
                from utils.rag import make_rag_zip_from_zip_path
                rag_path = get_zip_path(file_id)
                if rag_path and rag_path.exists():
                    rag_bytes = make_rag_zip_from_zip_path(
                        rag_path,
                        api_key,
                        chunk_size=body.chunk_size,
                        chunk_overlap=body.chunk_overlap,
                    )
                    rag_filename = f"faiss_db_{file_id[:8]}.zip"
                    rag_file_id = save_zip(rag_bytes, rag_filename, folder=FOLDER_RAG)
                    item["rag_file_id"] = rag_file_id
                    item["rag_filename"] = rag_filename
                    item["rag_download_url"] = f"{base_url}/zip/download/{rag_file_id}"
            except ValueError as e:
                item["rag_error"] = str(e)
            except Exception as e:
                item["rag_error"] = str(e)
        outputs.append(item)

    return {"source_file_id": body.file_id, "outputs": outputs}


@router.post("/generate-question")
def generate_question_api(body: GenerateQuestionRequest):
    """
    依使用者選擇的 RAG ZIP（file_id）載入向量庫 → 檢索 Context → 呼叫 GPT-4o 生成題目。
    需傳入 openai_api_key、題型 qtype、難度 level。回傳 JSON：question_content, hint, target_filename。
    """
    path = get_zip_path(body.file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該 RAG ZIP，請確認 file_id（可為 pack 回傳的 rag_file_id）")

    api_key = (body.openai_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 openai_api_key")

    try:
        from utils.question_gen import generate_question
        result = generate_question(
            path,
            api_key=api_key,
            qtype=body.qtype,
            level=body.level,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
