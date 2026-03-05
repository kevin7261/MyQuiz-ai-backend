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
    llm_api_key: str | None = None,
    system_prompt_instruction: str | None = None,
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
        "applied": False,
        "deleted": False,
        "updated_at": _now_utc_iso(),
    }
    if person_id is not None:
        row["person_id"] = person_id
    if name is not None:
        row["name"] = name
    if course_name is not None:
        row["course_name"] = course_name
    if llm_api_key is not None:
        row["llm_api_key"] = llm_api_key
    if system_prompt_instruction is not None:
        row["system_prompt_instruction"] = system_prompt_instruction
    return row


def _rag_table_select(select_columns: str = "*", exclude_deleted: bool = False) -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。（表名為 public.Rag）exclude_deleted=True 時僅回傳 deleted=False。"""
    supabase = get_supabase()
    q = supabase.table("Rag").select(select_columns)
    if exclude_deleted:
        q = q.eq("deleted", False)
    resp = q.execute()
    return resp.data or []


def _quizzes_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id 查詢 Quiz 表，回傳 rag_id -> list of quiz 列。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Quiz").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            out.setdefault(rid, []).append(row)
    return out


def _answers_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id 查詢 Answer 表，回傳 rag_id -> list of answer 列（與資料庫欄位一致）。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Answer").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            out.setdefault(rid, []).append(row)
    return out


class ListRagResponse(BaseModel):
    """GET /rag/rags 回應：Rag 表全部資料（含 applied、llm_api_key 等欄位），每筆另含關聯的 quizzes（每筆 quiz 帶 answers）、以及頂層 answers。"""
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
    system_prompt_instruction: str = ""  # 出題系統指令，寫入 Rag 表 system_prompt_instruction 欄位


def _resolve_person_id(form_person_id: str | None, x_person_id: str | None) -> str | None:
    """優先 Form person_id，若無則 Header X-Person-Id。回傳非空字串或 None。"""
    for raw in (form_person_id, x_person_id):
        if raw is not None and raw.strip():
            return raw.strip()
    return None


@router.get("/rags", response_model=ListRagResponse)
def list_rag(
    x_llm_api_key: str | None = Header(None, alias="X-LLM-Api-Key", description="LLM/OpenAI API Key（可選，與頁面 OpenAI API Key 對應）"),
):
    """
    列出 Rag 表內容，僅回傳 deleted=False 的資料；每筆 Rag 含表上所有欄位（含 applied、llm_api_key），並帶關聯的 Quiz（quizzes）與 Answer（answers）。
    關聯方式：quizzes 下每筆 quiz 帶 answers（依 quiz_id 關聯）；頂層 answers 為該 rag 下全部答案的扁平列表。
    LLM/OpenAI API Key 可選，由 Header X-LLM-Api-Key 傳入（與前端 OpenAI API Key 欄位對應）。
    """
    try:
        data = _rag_table_select(RAG_SELECT_ALL, exclude_deleted=True)
        rag_ids = []
        for row in data:
            rid = row.get("rag_id")
            if rid is not None:
                try:
                    rag_ids.append(int(rid))
                except (TypeError, ValueError):
                    pass
        rag_ids = list(dict.fromkeys(rag_ids))  # 去重且保持順序
        quizzes_by_rag = _quizzes_by_rag_id(rag_ids)
        answers_by_rag = _answers_by_rag_id(rag_ids)
        # 依 quiz_id 彙總 answers，供每筆 quiz 帶關聯的 answers
        answers_by_quiz_id: dict[int, list[dict]] = {}
        for rid in rag_ids:
            for a in answers_by_rag.get(rid, []):
                qid = a.get("quiz_id")
                if qid is not None:
                    try:
                        qid_int = int(qid)
                        answers_by_quiz_id.setdefault(qid_int, []).append(a)
                    except (TypeError, ValueError):
                        pass
        for row in data:
            rid = row.get("rag_id")
            rid_int = int(rid) if rid is not None else None
            row_quizzes = quizzes_by_rag.get(rid_int, []) if rid_int is not None else []
            for quiz in row_quizzes:
                qid = quiz.get("quiz_id")
                qid_int = int(qid) if qid is not None else None
                quiz["answers"] = answers_by_quiz_id.get(qid_int, []) if qid_int is not None else []
            row["quizzes"] = row_quizzes
            row["answers"] = answers_by_rag.get(rid_int, []) if rid_int is not None else []
            # 確保每筆都帶 llm_api_key（Rag 表欄位），供前端顯示／預填 API Key
            if "llm_api_key" not in row:
                row["llm_api_key"] = None
            # 別名 apikey，與 llm_api_key 同值，方便前端讀取
            row["apikey"] = row.get("llm_api_key") or ""
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    person_id: str | None = Form(None, description="寫入 Rag 表與儲存路徑的 person_id"),
    name: str | None = Form(None, description="寫入 Rag 表的 name 欄位；未傳則用上傳檔名（去掉 .zip）"),
    llm_api_key: str = Form(..., description="寫入 Rag 表的 llm_api_key 欄位；必填"),
    system_prompt_instruction: str | None = Form(None, description="寫入 Rag 表的 system_prompt_instruction 欄位；可選，出題系統指令"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    Upload Zip：傳入 person_id、ZIP 檔案與 llm_api_key，後端生成 file_id、上傳 ZIP 並新增 Rag 表一筆資料。
    person_id 可由 Form 或 Header X-Person-Id 傳入。name 可選；未傳則以檔名（去掉 .zip）寫入 Rag.name。
    course_name 不由上傳傳入，改以檔名（去掉 .zip）寫入 Rag.course_name。llm_api_key 必填。system_prompt_instruction 可選，寫入 Rag 表。
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

    # course_name 以檔名（去掉 .zip）寫入，不由上傳傳入
    course_name_val = Path(file.filename).stem if file.filename else None
    llm_key = (llm_api_key or "").strip()
    if not llm_key:
        raise HTTPException(status_code=400, detail="請傳入 llm_api_key（必填）")
    sys_prompt = (system_prompt_instruction or "").strip() or None
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .insert(_rag_default_row(file_id, person_id=resolved_person_id, name=rag_name, course_name=course_name_val, llm_api_key=llm_key, system_prompt_instruction=sys_prompt, file_metadata={}))
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
            "course_name": course_name_val,
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
            "rag_name": file_id,
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
                save_zip(
                    rag_bytes,
                    f"{file_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_file_id=body.file_id,
                    file_id=rag_file_id,
                )
            else:
                item["rag_error"] = "找不到 repack ZIP 路徑"
        except ValueError as e:
            item["rag_error"] = str(e)
        except Exception as e:
            item["rag_error"] = str(e)
        outputs.append(item)

    response = {"source_file_id": body.file_id, "rag_list": body.rag_list, "outputs": outputs}
    try:
        supabase = get_supabase()
        update_payload = {
            "rag_list": body.rag_list,
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
            "system_prompt_instruction": body.system_prompt_instruction or "",
            "updated_at": _now_utc_iso(),
        }
        supabase.table("Rag").update(update_payload).eq("file_id", body.file_id).execute()
    except Exception:
        pass
    return response


class UpdateRagNameRequest(BaseModel):
    """更新 Rag 的 name 欄位。"""
    name: str  # 新名稱，可為空字串


class UpdateRagLlmApiKeyRequest(BaseModel):
    """更新 Rag 的 llm_api_key 欄位。"""
    llm_api_key: str  # 新 API key，可為空字串


def _set_applied_only_for_file_id(supabase, pid: str, fid: str, extra_target_fields: dict | None = None) -> None:
    """將同一 person_id 下該 file_id 的 Rag 設為 applied=true，其餘皆設為 applied=false。extra_target_fields 會一併寫入該筆。"""
    now = _now_utc_iso()
    supabase.table("Rag").update({"applied": False, "updated_at": now}).eq("person_id", pid).neq("file_id", fid).execute()
    payload = {"applied": True, "updated_at": now}
    if extra_target_fields:
        payload.update(extra_target_fields)
    supabase.table("Rag").update(payload).eq("file_id", fid).eq("person_id", pid).execute()


@router.patch("/name/{file_id}", status_code=200)
def update_rag_name(
    file_id: str = PathParam(..., description="要更新 name 的 Rag 的 file_id"),
    body: UpdateRagNameRequest = ...,
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    PATCH /rag/name/{file_id}，body 傳 { "name": "新名稱" }，person_id 請帶 Header X-Person-Id。
    更新 Rag 表該筆的 name、updated_at，並將該筆 applied 設為 true、同 person_id 下其餘皆設為 false。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (file_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 file_id")
    try:
        supabase = get_supabase()
        r = supabase.table("Rag").select("rag_id").eq("file_id", fid).eq("person_id", pid).execute()
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 file_id 的 Rag 資料")
        _set_applied_only_for_file_id(supabase, pid, fid, extra_target_fields={"name": body.name})
        return {"message": "已更新 name 並設為目前套用", "file_id": fid, "name": body.name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/applied/{file_id}", status_code=200)
def set_rag_applied(
    file_id: str = PathParam(..., description="要設為套用中的 Rag 的 file_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    PATCH /rag/applied/{file_id}，person_id 請帶 Header X-Person-Id。
    將該 file_id 的 Rag 設為 applied=true，同 person_id 下其餘 Rag 皆設為 applied=false。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (file_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 file_id")
    try:
        supabase = get_supabase()
        r = supabase.table("Rag").select("rag_id").eq("file_id", fid).eq("person_id", pid).eq("deleted", False).execute()
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 file_id 的 Rag 資料")
        _set_applied_only_for_file_id(supabase, pid, fid)
        return {"message": "已設為目前套用", "file_id": fid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/llm_api_key/{file_id}", status_code=200)
def update_rag_llm_api_key(
    file_id: str = PathParam(..., description="要更新 llm_api_key 的 Rag 的 file_id"),
    body: UpdateRagLlmApiKeyRequest = ...,
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    PATCH /rag/llm_api_key/{file_id}，body 傳 { "llm_api_key": "sk-..." }，person_id 請帶 Header X-Person-Id。
    僅更新 Rag 表該筆的 llm_api_key 與 updated_at。
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
            .update({"llm_api_key": body.llm_api_key, "updated_at": _now_utc_iso()})
            .eq("file_id", fid)
            .eq("person_id", pid)
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 file_id 的 Rag 資料")
        return {"message": "已更新 llm_api_key", "file_id": fid}
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
