"""ZIP 相關 API 路由。"""

import io
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header
from pydantic import BaseModel

from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
)
from utils.storage import (
    save_zip,
    get_zip_path,
    get_zip_path_by_person,
    generate_file_id,
    FOLDER_UPLOAD,
    FOLDER_REPACK,
    FOLDER_RAG,
)
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/rag", tags=["rag"])

# Rag 表列出全部資料時選取欄位（用 * 回傳全部欄位）
RAG_SELECT_ALL = "*"


def _rag_default_row(
    file_id: str,
    *,
    person_id: str | None = None,
    file_metadata: Any = None,
) -> dict[str, Any]:
    """Rag 表一筆新增時的預設欄位，供 upload_zip 使用。"""
    row: dict[str, Any] = {
        "file_id": file_id,
        "file_metadata": file_metadata,
        "rag_list": "",
        "rag_metadata": None,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "deleted": False,
    }
    if person_id is not None:
        row["person_id"] = person_id
    return row


def _rag_table_select(select_columns: str = "*") -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。（表名為 public.Rag）"""
    supabase = get_supabase()
    resp = supabase.table("Rag").select(select_columns).execute()
    return resp.data or []


def _rag_table_select_by_file_id(file_id: str, select_columns: str = "*") -> list[dict]:
    """查詢 Rag 表指定 file_id 的列。（表名為 public.Rag）"""
    supabase = get_supabase()
    resp = supabase.table("Rag").select(select_columns).eq("file_id", file_id).execute()
    return resp.data or []


class ListRagResponse(BaseModel):
    """GET /rag/rags 回應：Rag 表全部資料。"""
    rags: list[dict]
    count: int


class PackRequest(BaseModel):
    """指定先前上傳的 ZIP（file_id）與要打包的資料夾規則。ZIP 路徑為 {person_id}/{file_id}/upload。會 update Rag 表該 file_id 的 rag_list、rag_metadata、chunk_size、chunk_overlap。"""
    file_id: str
    person_id: str  # 與 upload-zip 一致，上傳 ZIP 所在路徑的 person_id
    rag_list: str  # 寫入 Rag 表 rag_list 欄位；例："220222+220301" 或 "220222,220301+220302"（逗號=多個 ZIP，加號=同一 ZIP 多資料夾）
    openai_api_key: str  # 用於 Embedding，必填（一律做成 RAG ZIP）
    chunk_size: int = 1000  # 寫入 Rag 表 chunk_size 欄位
    chunk_overlap: int = 200  # 寫入 Rag 表 chunk_overlap 欄位


class GenerateQuestionRequest(BaseModel):
    """指定 RAG ZIP 的來源 file_id（upload-zip 回傳）、rag_name（rag_list 的某一段，如 220222_220301）與出題參數。"""
    file_id: str  # upload-zip 回傳的 source file_id
    rag_name: str  # create-rag 時 rag_list 某一段的 stem，如 220222_220301；程式會以 {rag_name}_rag 查找 rag 檔案
    openai_api_key: str  # 用於 GPT-4o 出題，不從環境變數讀取
    qtype: str  # 題型
    level: str  # 難度


def _resolve_person_id(form_person_id: str | None, x_person_id: str | None) -> str | None:
    """優先 Form person_id，若無則 Header X-Person-Id。回傳非空字串或 None。"""
    for raw in (form_person_id, x_person_id):
        if raw is not None and raw.strip():
            return raw.strip()
    return None


@router.get("/rags", response_model=ListRagResponse)
def list_rag():
    """
    列出 Rag 表全部內容（與 GET /users 一樣回傳全部資料）。
    """
    try:
        data = _rag_table_select(RAG_SELECT_ALL)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    person_id: str | None = Form(None, description="寫入 Rag 表與儲存路徑的 person_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    Upload Zip：傳入 person_id 與 ZIP 檔案，後端生成 file_id、上傳 ZIP 並新增 Rag 表一筆資料。
    person_id 可由 Form 或 Header X-Person-Id 傳入。
    回傳 rag_id、file_id（後端生成）、created_at、filename、second_folders（並寫入 file_metadata）。
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

    resolved_person_id = _resolve_person_id(person_id, x_person_id)
    if not resolved_person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id（Form 或 Header X-Person-Id）")

    file_id = generate_file_id(resolved_person_id)
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

    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .insert(_rag_default_row(file_id, person_id=resolved_person_id, file_metadata={}))
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=500, detail="新增 Rag 失敗")
        row = r.data[0]
        # 完整回傳內容寫入 file_metadata
        file_metadata = {
            "rag_id": row["rag_id"],
            "file_id": row["file_id"],
            "created_at": row["created_at"],
            "filename": file.filename,
            "second_folders": folders,
        }
        supabase.table("Rag").update({"file_metadata": file_metadata}).eq("file_id", file_id).execute()
        return file_metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-rag")
def create_rag(body: PackRequest):
    """
    依先前上傳的 ZIP（file_id）與 rag_list 字串，抽出指定 6 位數資料夾重新壓成 ZIP 並存到後端。
    ZIP 檔位置為 {person_id}/{file_id}/upload（與 upload-zip 一致），body 需傳入 person_id。
    rag_list 寫入 Rag 表；格式：逗號分隔多個輸出檔，加號為同一檔內多個資料夾。
    一律做成 RAG（FAISS）ZIP，需傳入 openai_api_key。回傳內容完整寫入 Rag 表 rag_metadata，並 update chunk_size、chunk_overlap。
    """
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")

    path = get_zip_path(body.file_id) or get_zip_path_by_person(pid, body.file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該上傳的 ZIP，請先上傳或確認 file_id、person_id")

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.rag_list)
    if not packed:
        raise HTTPException(status_code=400, detail="rag_list 為空或格式錯誤，例：220222+220301")

    api_key = (body.openai_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="請傳入 openai_api_key")

    outputs = []
    for zip_bytes, filename in packed:
        # 用 rag_list 衍生的檔名做 file_id（如 220222_220301.zip -> 220222_220301），不再生成 UUID
        repack_file_id = Path(filename).stem if filename else None
        if not repack_file_id or "/" in repack_file_id or "\\" in repack_file_id:
            repack_file_id = str(uuid.uuid4())
        file_id = save_zip(
            zip_bytes,
            filename,
            folder=FOLDER_REPACK,
            person_id=pid,
            parent_file_id=body.file_id,
            file_id=repack_file_id,
        )
        item = {
            "file_id": file_id,
            "filename": filename,
        }
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
                # rag 檔名也依 rag_list，file_id 加 _rag 以區分 repack
                rag_file_id = f"{file_id}_rag"
                rag_filename = f"{file_id}.zip"
                save_zip(
                    rag_bytes,
                    rag_filename,
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_file_id=body.file_id,
                    file_id=rag_file_id,
                )
                item["rag_file_id"] = rag_file_id
                item["rag_filename"] = rag_filename
            else:
                item["rag_error"] = "找不到 repack ZIP 路徑"
        except ValueError as e:
            item["rag_error"] = str(e)
        except Exception as e:
            item["rag_error"] = str(e)
        outputs.append(item)

    response = {"source_file_id": body.file_id, "outputs": outputs}
    try:
        supabase = get_supabase()
        supabase.table("Rag").update({
            "rag_list": body.rag_list,
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
        }).eq("file_id", body.file_id).execute()
    except Exception:
        pass
    return response


@router.post("/generate-question")
def generate_question_api(body: GenerateQuestionRequest):
    """
    傳入 file_id（upload-zip 的 source file_id）與 rag_name（如 220222_220301），程式自動組出 rag_file_id={rag_name}_rag 並查找 RAG ZIP。
    再呼叫 GPT-4o 生成題目，需傳入 openai_api_key、題型 qtype、難度 level。回傳 JSON：question_content, hint, target_filename。
    """
    rag_name = (body.rag_name or "").strip()
    if not rag_name:
        raise HTTPException(status_code=400, detail="請傳入 rag_name（如 220222_220301）")
    rag_file_id = f"{rag_name}_rag"
    path = get_zip_path(rag_file_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到 RAG ZIP，請確認 file_id={body.file_id}、rag_name={rag_name}（rag_file_id={rag_file_id}）")

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
