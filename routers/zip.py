"""ZIP 相關 API 路由。"""

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc_iso() -> str:
    """回傳目前 UTC 時間的 ISO 字串，供 Rag 表 updated_at 使用。"""
    return datetime.now(timezone.utc).isoformat()

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Path as PathParam
from fastapi.responses import Response
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
    delete_file_folder,
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
    name: str | None = None,
    course_name: str | None = None,
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
        "updated_at": _now_utc_iso(),
    }
    if person_id is not None:
        row["person_id"] = person_id
    if name is not None:
        row["name"] = name
    if course_name is not None:
        row["course_name"] = course_name
    return row


def _rag_table_select(select_columns: str = "*", exclude_deleted: bool = False) -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。（表名為 public.Rag）exclude_deleted=True 時僅回傳 deleted=False。"""
    supabase = get_supabase()
    q = supabase.table("Rag").select(select_columns)
    if exclude_deleted:
        q = q.eq("deleted", False)
    resp = q.execute()
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
    """指定先前上傳的 ZIP（file_id）與要打包的資料夾規則。ZIP 路徑為 {person_id}/{file_id}/upload。會 update Rag 表該 file_id 的 rag_list、rag_metadata、chunk_size、chunk_overlap、system_prompt_instruction。"""
    file_id: str
    person_id: str  # 與 upload-zip 一致，上傳 ZIP 所在路徑的 person_id
    rag_list: str  # 寫入 Rag 表 rag_list 欄位；例："220222+220301" 或 "220222,220301+220302"（逗號=多個 ZIP，加號=同一 ZIP 多資料夾）
    openai_api_key: str  # 用於 Embedding，必填（一律做成 RAG ZIP）
    chunk_size: int = 1000  # 寫入 Rag 表 chunk_size 欄位
    chunk_overlap: int = 200  # 寫入 Rag 表 chunk_overlap 欄位
    system_prompt_instruction: str | None = None  # 寫入 Rag 表 system_prompt_instruction 欄位（選填），供出題等使用


class GenerateQuestionRequest(BaseModel):
    """指定 RAG ZIP 的來源 file_id（upload-zip 回傳）、rag_name（rag_list 的某一段，如 220222_220301）與出題參數。"""
    file_id: str  # upload-zip 回傳的 source file_id（用於查 Rag 表的 system_prompt_instruction）
    rag_name: str  # create-rag-zip 時 rag_list 某一段的 stem，如 220222_220301；程式會以 {rag_name}_rag 查找 rag 檔案（回傳時作為選擇單元／壓縮檔名）
    openai_api_key: str  # 用於 GPT-4o 出題，不從環境變數讀取
    level: str  # 難度（回傳時一併帶回）
    system_prompt_instruction: str | None = None  # 出題系統指令（選填）；若未傳則使用 Rag 表的 system_prompt_instruction，若皆有則合併（回傳時一併帶回）
    course_name: str  # 課程名稱，會帶入出題 prompt 中


def _resolve_person_id(form_person_id: str | None, x_person_id: str | None) -> str | None:
    """優先 Form person_id，若無則 Header X-Person-Id。回傳非空字串或 None。"""
    for raw in (form_person_id, x_person_id):
        if raw is not None and raw.strip():
            return raw.strip()
    return None


@router.get("/rags", response_model=ListRagResponse)
def list_rag():
    """
    列出 Rag 表內容，僅回傳 deleted=False 的資料。
    """
    try:
        data = _rag_table_select(RAG_SELECT_ALL, exclude_deleted=True)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    person_id: str | None = Form(None, description="寫入 Rag 表與儲存路徑的 person_id"),
    name: str | None = Form(None, description="寫入 Rag 表的 name 欄位；未傳則用上傳檔名（去掉 .zip）"),
    course_name: str | None = Form(None, description="寫入 Rag 表的 course_name 欄位"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    Upload Zip：傳入 person_id 與 ZIP 檔案，後端生成 file_id、上傳 ZIP 並新增 Rag 表一筆資料。
    person_id 可由 Form 或 Header X-Person-Id 傳入。name、course_name 可選；name 未傳則以檔名（去掉 .zip）寫入 Rag.name。
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

    # Rag.name：優先 Form name，否則用上傳檔名（去掉 .zip）
    rag_name = (name or "").strip() if name is not None else None
    if not rag_name and file.filename:
        rag_name = Path(file.filename).stem or None

    course_name_val = (course_name or "").strip() if course_name is not None else None
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .insert(_rag_default_row(file_id, person_id=resolved_person_id, name=rag_name, course_name=course_name_val, file_metadata={}))
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
        supabase.table("Rag").update({"file_metadata": file_metadata, "updated_at": _now_utc_iso()}).eq("file_id", file_id).execute()
        return file_metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-rag-zip")
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
        update_payload = {
            "rag_list": body.rag_list,
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
            "updated_at": _now_utc_iso(),
        }
        if body.system_prompt_instruction is not None:
            update_payload["system_prompt_instruction"] = (body.system_prompt_instruction or "").strip() or None
        supabase.table("Rag").update(update_payload).eq("file_id", body.file_id).execute()
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

    # 合併 Rag 表的 system_prompt_instruction 與 request body 的 system_prompt_instruction
    rag_rows = _rag_table_select_by_file_id(body.file_id, select_columns="system_prompt_instruction")
    rag_system = (rag_rows[0].get("system_prompt_instruction") or "").strip() if rag_rows else ""
    body_system = (body.system_prompt_instruction or "").strip()
    if rag_system and body_system:
        system_prompt_instruction = f"{rag_system}\n{body_system}"
    else:
        system_prompt_instruction = rag_system or body_system
    if not system_prompt_instruction:
        raise HTTPException(status_code=400, detail="請傳入 system_prompt_instruction（request body），或於 Rag 表該 file_id 設定 system_prompt_instruction")

    course_name = (body.course_name or "").strip()
    if not course_name:
        raise HTTPException(status_code=400, detail="請傳入 course_name（課程名稱，必填）")

    try:
        from utils.question_gen import generate_question
        result = generate_question(
            path,
            api_key=api_key,
            level=body.level,
            system_prompt_instruction=system_prompt_instruction,
            course_name=course_name,
        )
        # 回傳加上 system_prompt_instruction、選擇單元（壓縮檔名）、難度
        result["system_prompt_instruction"] = system_prompt_instruction
        result["unit_filename"] = f"{rag_name}_rag.zip"  # 選擇單元（壓縮檔名）
        result["level"] = body.level
        # 明確以 UTF-8 回傳 JSON，避免 'ascii' codec can't encode 錯誤（題目/提示/答案含中文）
        body_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return Response(content=body_bytes, media_type="application/json; charset=utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateRagNameRequest(BaseModel):
    """更新 Rag 的 name 欄位。"""
    name: str  # 新名稱，可為空字串


@router.patch("/name/{file_id}", status_code=200)
def update_rag_name(
    file_id: str = PathParam(..., description="要更新 name 的 Rag 的 file_id"),
    body: UpdateRagNameRequest = ...,
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    PATCH /rag/name/{file_id}，body 傳 { "name": "新名稱" }，person_id 請帶 Header X-Person-Id。
    僅更新 Rag 表該筆的 name 與 updated_at。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (file_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 file_id")
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .update({"name": body.name, "updated_at": _now_utc_iso()})
            .eq("file_id", fid)
            .eq("person_id", pid)
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 file_id 的 Rag 資料")
        return {"message": "已更新 name", "file_id": fid, "name": body.name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _do_delete_rag_file(pid: str, fid: str):
    """共用：將 Rag 表該筆 deleted 設為 true 並刪除 storage 資料夾。"""
    try:
        supabase = get_supabase()
        supabase.table("Rag").update({"deleted": True, "updated_at": _now_utc_iso()}).eq("file_id", fid).eq("person_id", pid).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Rag 表失敗: {e}")
    try:
        folder_deleted = delete_file_folder(pid, fid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return folder_deleted


@router.post("/delete/{file_id}", status_code=200)
def delete_rag_file(
    file_id: str = PathParam(..., description="要刪除的 file_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    POST /rag/delete/{file_id}，person_id 請帶 Header X-Person-Id。
    軟刪除：將 Rag 表該筆 deleted 設為 true，並刪除 storage/{person_id}/{file_id}/ 整個資料夾。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (file_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 file_id")
    folder_deleted = _do_delete_rag_file(pid, fid)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "file_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }
