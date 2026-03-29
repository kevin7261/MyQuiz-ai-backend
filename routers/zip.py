"""
ZIP 與 RAG 相關 API 模組。
提供：
- GET /rag/tabs：列出 Rag 表（含 quizzes、answers）；query `local` 篩選 Rag.local，未傳時依連線是否本機判定；回傳依 created_at 舊→新
- GET /rag/tab/for-exam：依連線讀取 System_Setting（rag_localhost / rag_deploy）的 rag_id，回傳對應 Rag
- POST /rag/tab/create：建立一筆 Rag（可傳 local）
- PUT /rag/tab/tab-name：更新既有 Rag 的 tab_name（body：rag_id、tab_name；與 tab/create 回傳之 rag_id 相同）
- POST /rag/tab/upload-zip：上傳 ZIP
- POST /rag/tab/build-rag-zip：依 unit_list 打包並建 RAG
- POST /rag/tab/delete/{rag_tab_id}：軟刪除並刪除儲存
"""

# 引入 io 用於 BytesIO 等
import io
# 引入 logging 用於記錄錯誤
import logging
# 引入 uuid 用於產生 repack tab_id
import uuid
# 引入 zipfile 用於讀取 ZIP
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path
# 引入 Any 型別
from typing import Any

# 引入 FastAPI 的 APIRouter、HTTPException、UploadFile、File、Form、Header、PathParam、Query、Request
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Path as PathParam, Query, Request
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 引入 UTC 時間工具
from utils.datetime_utils import now_utc_iso
# 引入 to_json_safe 轉換 datetime 等
from utils.json_utils import to_json_safe
# 依 person_id 取得 LLM API Key
from utils.llm_api_key_utils import get_llm_api_key_for_person
# ZIP 工具：第二層資料夾、folder_map、repack
from utils.zip_utils import (
    get_second_level_folders_from_zip_file,
    build_folder_map,
    repack_tasks_to_zips,
)
# 儲存工具
from utils.zip_storage import (
    save_zip,
    get_zip_path,
    get_zip_path_by_person,
    delete_tab_folder,
    FOLDER_UPLOAD,
    FOLDER_REPACK,
    FOLDER_RAG,
)
# Supabase 客戶端
from utils.supabase_client import get_supabase
# 供測驗 RAG：System_Setting rag_id、本機判定
from utils.rag_exam_setting import fetch_exam_rag_id_from_settings, is_localhost_request

# 建立路由，前綴 /rag
router = APIRouter(prefix="/rag", tags=["rag"])

# Rag 表列出全部資料時選取欄位（用 * 回傳全部欄位）
RAG_SELECT_ALL = "*"


def _rag_default_row(
    rag_tab_id: str,
    *,
    tab_name: str | None = None,
    person_id: str | None = None,
    system_prompt_instruction: str | None = None,
    file_metadata: Any = None,
    local: bool = False,
) -> dict[str, Any]:
    """Rag 表一筆新增時的預設欄位；鍵順序同 public.Rag（rag_tab_id→…→updated_at，不含 rag_id/created_at）。"""
    row: dict[str, Any] = {
        "rag_tab_id": rag_tab_id,
        "tab_name": tab_name if tab_name is not None else "",
        "person_id": person_id if person_id is not None else "",
        "file_metadata": file_metadata,
        "unit_list": "",
    }
    if system_prompt_instruction is not None:
        row["system_prompt_instruction"] = system_prompt_instruction
    row["rag_metadata"] = None
    row["chunk_size"] = 1000
    row["chunk_overlap"] = 200
    row["local"] = local
    row["deleted"] = False
    row["updated_at"] = now_utc_iso()
    return row


def _rag_table_select(
    select_columns: str = "*",
    exclude_deleted: bool = False,
    *,
    local_match: bool | None = None,
) -> list[dict]:
    """查詢 Rag 表全部列。回傳 list of dict。（表名為 public.Rag）exclude_deleted=True 時僅回傳 deleted=False。local_match 若指定則僅回傳 Rag.local 與其相符的列。依 created_at 升序（舊→新）。"""
    supabase = get_supabase()
    q = supabase.table("Rag").select(select_columns)
    if exclude_deleted:
        q = q.eq("deleted", False)
    if local_match is not None:
        q = q.eq("local", local_match)
    q = q.order("created_at", desc=False)
    resp = q.execute()
    return resp.data or []


def _quizzes_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id 查詢 Rag_Quiz 表，回傳 rag_id -> list of quiz 列。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Rag_Quiz").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            out.setdefault(rid, []).append(row)
    return out


def _answers_by_rag_id(rag_ids: list[int]) -> dict[int, list[dict]]:
    """依 rag_id 查詢 Rag_Answer 表，回傳 rag_id -> list of answer 列（與資料庫欄位一致）。"""
    if not rag_ids:
        return {}
    supabase = get_supabase()
    resp = supabase.table("Rag_Answer").select("*").in_("rag_id", rag_ids).execute()
    rows = resp.data or []
    out: dict[int, list[dict]] = {rid: [] for rid in rag_ids}
    for row in rows:
        rid = row.get("rag_id")
        if rid is not None:
            out.setdefault(rid, []).append(row)
    return out


class ListRagResponse(BaseModel):
    """GET /rag/tabs 回應：Rag 表全部欄位，每筆另含關聯的 Rag_Quiz（quizzes，每題帶一筆 answer）、以及頂層 Rag_Answer（answers）。"""
    rags: list[dict]
    count: int


class CreateRagRequest(BaseModel):
    """POST /rag/tab/create：欄位順序同 public.Rag（rag_tab_id, tab_name, person_id, local；其餘欄位於 tab/upload-zip、tab/build-rag-zip 寫入）。"""
    rag_tab_id: str = Field(..., description="Rag 的 tab 識別，對應 Rag 表 rag_tab_id 欄位")
    tab_name: str = Field(..., description="Rag 顯示名稱，寫入 Rag 表 tab_name 欄位")
    person_id: str = Field(..., description="使用者/路徑識別")
    local: bool = Field(False, description="是否為本機 RAG，寫入 Rag 表 local 欄位")


class UpdateRagUnitNameRequest(BaseModel):
    """PUT /rag/tab/tab-name：請求僅含 rag_id（主鍵）、tab_name；勿傳 rag_tab_id。"""
    rag_id: int = Field(..., description="Rag 表主鍵（整數），與 POST /rag/tab/create 回傳之 rag_id 相同；辨識請用 rag_id，非 rag_tab_id")
    tab_name: str = Field(..., description="新的顯示名稱，寫入 Rag 表 tab_name 欄位")


class PackRequest(BaseModel):
    """欄位順序對應 public.Rag 中本請求會更新的區段：rag_tab_id, person_id, unit_list, system_prompt_instruction, chunk_size, chunk_overlap（另寫 rag_metadata；比對 person_id + rag_tab_id 更新）。"""
    rag_tab_id: str
    person_id: str  # 與 tab/upload-zip 一致，上傳 ZIP 所在路徑的 person_id
    unit_list: str  # 寫入 Rag 表 unit_list 欄位；例："220222+220301" 或 "220222,220301+220302"（逗號=多個 ZIP，加號=同一 ZIP 多資料夾）
    system_prompt_instruction: str = ""  # 出題系統指令，寫入 Rag 表 system_prompt_instruction 欄位
    chunk_size: int = 1000  # 寫入 Rag 表 chunk_size 欄位
    chunk_overlap: int = 200  # 寫入 Rag 表 chunk_overlap 欄位


@router.get("/tabs", response_model=ListRagResponse)
def list_rag(
    request: Request,
    local: bool | None = Query(
        None,
        description="僅回傳 Rag.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false（與前端在本機開發傳 true、正式環境傳 false 一致）",
    ),
):
    """
    列出 Rag 表內容，僅回傳 deleted=False 的資料，且 Rag.local 須與 query `local` 相符（未傳 `local` 時依連線是否本機自動判定）。
    回傳列依 created_at 由舊到新排序。
    每筆 Rag 含表上所有欄位，並帶關聯的 Rag_Quiz（quizzes）與 Rag_Answer（answers）。
    關聯方式：quizzes 下每筆 quiz 帶 answers（依 rag_quiz_id 關聯，每題僅一筆）；頂層 answers 為該 rag 下全部答案的扁平列表。
    LLM API Key 依 person_id 從 User 表取得。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _rag_table_select(RAG_SELECT_ALL, exclude_deleted=True, local_match=local_filter)
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
        # 依 rag_quiz_id 彙總 answers，供每筆 quiz 帶關聯的 answers
        answers_by_rag_quiz_id: dict[int, list[dict]] = {}
        for rid in rag_ids:
            for a in answers_by_rag.get(rid, []):
                qid = a.get("rag_quiz_id")
                if qid is not None:
                    try:
                        qid_int = int(qid)
                        answers_by_rag_quiz_id.setdefault(qid_int, []).append(a)
                    except (TypeError, ValueError):
                        pass
        for row in data:
            rid = row.get("rag_id")
            rid_int = int(rid) if rid is not None else None
            row_quizzes = quizzes_by_rag.get(rid_int, []) if rid_int is not None else []
            for quiz in row_quizzes:
                qid = quiz.get("rag_quiz_id")
                qid_int = int(qid) if qid is not None else None
                # 每題 quiz 只帶一筆 answer（取第一筆）
                raw_answers = (answers_by_rag_quiz_id.get(qid_int, []) or []) if qid_int is not None else []
                quiz["answers"] = raw_answers[:1]
            row["quizzes"] = row_quizzes
            row["answers"] = answers_by_rag.get(rid_int, []) if rid_int is not None else []
        # 轉成可 JSON 序列化（Supabase 的 datetime 等），避免 500
        data = to_json_safe(data)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        logging.exception("GET /rag/tabs 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag 失敗: {e!s}")


@router.get("/tab/for-exam")
def get_for_exam_rag(request: Request):
    """
    依連線讀取 System_Setting（本機：rag_localhost，否則 rag_deploy）的 value 作為 rag_id，取得該筆 Rag（deleted=false）。
    回傳格式與 POST /rag/tab/build-rag-zip 一致，並多帶 rag_id、rag_tab_id、llm_api_key、system_prompt_instruction。
    """
    try:
        supabase = get_supabase()
        _, rag_id = fetch_exam_rag_id_from_settings(supabase, request)
        if rag_id is None or rag_id <= 0:
            return {
                "source_rag_tab_id": None,
                "unit_list": None,
                "outputs": [],
                "rag_id": None,
                "rag_tab_id": None,
                "llm_api_key": None,
                "system_prompt_instruction": None,
            }
        resp = (
            supabase.table("Rag")
            .select(RAG_SELECT_ALL)
            .eq("rag_id", rag_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        data = resp.data or []
        if len(data) == 0:
            return {
                "source_rag_tab_id": None,
                "unit_list": None,
                "outputs": [],
                "rag_id": None,
                "rag_tab_id": None,
                "llm_api_key": None,
                "system_prompt_instruction": None,
            }
        row = data[0]
        meta = row.get("rag_metadata")
        extra = {
            "rag_id": row.get("rag_id"),
            "rag_tab_id": row.get("rag_tab_id"),
            "llm_api_key": row.get("llm_api_key"),
            "system_prompt_instruction": row.get("system_prompt_instruction"),
        }
        if isinstance(meta, dict) and "source_rag_tab_id" in meta and "outputs" in meta:
            ul = meta.get("unit_list")
            if ul is None or ul == "":
                ul = meta.get("rag_list")  # 舊版 rag_metadata 內鍵名
            return to_json_safe({
                "source_rag_tab_id": meta.get("source_rag_tab_id"),
                "unit_list": ul,
                "outputs": meta.get("outputs", []),
                **extra,
            })
        row_ul = row.get("unit_list")
        if row_ul is None or row_ul == "":
            row_ul = row.get("rag_list") or ""
        return to_json_safe({
            "source_rag_tab_id": row.get("rag_tab_id"),
            "unit_list": row_ul,
            "outputs": (meta or {}).get("outputs", []) if isinstance(meta, dict) else [],
            **extra,
        })
    except Exception as e:
        logging.exception("GET /rag/tab/for-exam 錯誤")
        raise HTTPException(
            status_code=500,
            detail=f"取得供測驗 RAG 失敗: {e!s}",
        )


@router.post("/tab/create")
def create_unit(body: CreateRagRequest):
    """
    只建立一筆 Rag 資料，接受 rag_tab_id、person_id、tab_name（必填）、local（選填，預設 false）。system_prompt_instruction 請在 POST /rag/tab/build-rag-zip 傳入。
    回傳新增的 rag_id、rag_tab_id、person_id、tab_name、local、created_at。
    """
    fid = (body.rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        r = (
            supabase.table("Rag")
            .insert(
                _rag_default_row(
                    fid,
                    tab_name=tab_name,
                    person_id=pid,
                    file_metadata=None,
                    local=body.local,
                )
            )
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=500, detail="新增 Rag 失敗")
        row = r.data[0]
        return {
            "rag_id": row["rag_id"],
            "rag_tab_id": row["rag_tab_id"],
            "tab_name": row.get("tab_name"),
            "person_id": row.get("person_id"),
            "local": row.get("local"),
            "created_at": row.get("created_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tab/tab-name")
def update_unit_tab_name(body: UpdateRagUnitNameRequest):
    """
    更新既有 Rag 的 tab_name。以 rag_id（Rag 主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_id、rag_tab_id、person_id、tab_name、updated_at。
    """
    if body.rag_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 rag_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Rag")
            .select("rag_id, rag_tab_id, person_id")
            .eq("rag_id", body.rag_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_id 的 Rag 資料，或已刪除")
        row = sel.data[0]
        fid = row.get("rag_tab_id")
        pid = row.get("person_id")
        ts = now_utc_iso()
        supabase.table("Rag").update({"tab_name": tab_name, "updated_at": ts}).eq("rag_id", body.rag_id).eq("deleted", False).execute()
        return {
            "rag_id": body.rag_id,
            "rag_tab_id": fid,
            "person_id": pid,
            "tab_name": tab_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tab/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    rag_tab_id: str = Form(..., description="對應 tab/create 建立的 rag_tab_id（Rag 表 rag_tab_id），ZIP 會存於此路徑"),
    person_id: str = Form(..., description="寫入儲存路徑的 person_id，需與 tab/create 一致"),
):
    """
    Upload Zip：只做上傳並寫入資料庫。需先以 tab/create 建立該 rag_tab_id 的 Rag 資料。
    傳入 rag_tab_id（tab/create 的 rag_tab_id）、ZIP 檔案與 person_id（Form 必填）。
    出題/評分時由後端依 person_id 從 User 表取得 LLM API Key。
    會更新該筆 Rag 的 file_metadata（filename、second_folders 等）。
    回傳 file_metadata。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    fid = (rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            folders = get_second_level_folders_from_zip_file(zip_ref)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="無法讀取 ZIP 檔案")

    resolved_person_id = (person_id or "").strip()
    if not resolved_person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id")

    try:
        supabase = get_supabase()
        r = supabase.table("Rag").select("rag_id, created_at").eq("rag_tab_id", fid).eq("person_id", resolved_person_id).execute()
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_tab_id 的 Rag 資料，請先呼叫 POST /rag/tab/create 建立")
        row = r.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        save_zip(
            contents,
            file.filename,
            folder=FOLDER_UPLOAD,
            person_id=resolved_person_id,
            tab_id=fid,  # storage 內部仍用 tab_id 路徑
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_metadata = {
        "rag_id": row["rag_id"],
        "rag_tab_id": fid,
        "created_at": row["created_at"],
        "filename": file.filename,
        "second_folders": folders,
    }
    update_payload: dict[str, Any] = {
        "file_metadata": file_metadata,
        "updated_at": now_utc_iso(),
    }
    # llm_api_key 不寫入 Rag 表（該表無此欄位）；依 person_id 從 User 表取得
    try:
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", fid).eq("person_id", resolved_person_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return file_metadata


@router.post("/tab/build-rag-zip")
def build_rag_zip(body: PackRequest):
    """
    依先前上傳的 ZIP（rag_tab_id）與 unit_list 字串，抽出指定資料夾名稱（路徑上任一層目錄名皆可，含 6 位數週次）重新壓成 ZIP 並存到後端。
    ZIP 檔位置為 {person_id}/{rag_tab_id}/upload（與 tab/upload-zip 一致），body 需傳入 person_id。
    unit_list 寫入 Rag 表；格式：逗號分隔多個輸出檔，加號為同一檔內多個資料夾。
    一律做成 RAG（FAISS）ZIP；LLM API Key 依 person_id 從 User 表取得。回傳內容完整寫入 Rag 表 rag_metadata，並 update chunk_size、chunk_overlap。
    """
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")

    path = get_zip_path(body.rag_tab_id) or get_zip_path_by_person(pid, body.rag_tab_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到該上傳的 ZIP，請先上傳或確認 rag_tab_id、person_id")

    try:
        try:
            with zipfile.ZipFile(path, "r") as z:
                folder_map = build_folder_map(z)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

        packed = repack_tasks_to_zips(path, folder_map, body.unit_list)
        if not packed:
            raise HTTPException(status_code=400, detail="unit_list 為空或格式錯誤，例：220222+220301")

        api_key = get_llm_api_key_for_person(pid)
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
            )

        outputs = []
        for zip_bytes, filename in packed:
            # 用 unit_list 衍生的檔名做 tab_id（如 220222_220301.zip -> 220222_220301），不再生成 UUID
            repack_tab_id = Path(filename).stem if filename else None
            if not repack_tab_id or "/" in repack_tab_id or "\\" in repack_tab_id:
                repack_tab_id = str(uuid.uuid4())
            tab_id = save_zip(
                zip_bytes,
                filename,
                folder=FOLDER_REPACK,
                person_id=pid,
                parent_tab_id=body.rag_tab_id,
                tab_id=repack_tab_id,
            )
            item = {
                "unit_name": tab_id,
                "filename": filename,
            }
            try:
                from utils.rag_faiss_zip import make_rag_zip_from_zip_path
                rag_path = get_zip_path(tab_id)
                if rag_path and rag_path.exists():
                    try:
                        rag_bytes = make_rag_zip_from_zip_path(
                            rag_path,
                            api_key,
                            chunk_size=body.chunk_size,
                            chunk_overlap=body.chunk_overlap,
                        )
                    finally:
                        try:
                            rag_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    # rag 檔名也依 unit_list，tab_id 加 _rag 以區分 repack
                    rag_tab_id = f"{tab_id}_rag"
                    save_zip(
                        rag_bytes,
                        f"{tab_id}.zip",
                        folder=FOLDER_RAG,
                        person_id=pid,
                        parent_tab_id=body.rag_tab_id,
                        tab_id=rag_tab_id,
                    )
                else:
                    item["rag_error"] = "找不到 repack ZIP 路徑"
            except ValueError as e:
                item["rag_error"] = str(e)
            except Exception as e:
                item["rag_error"] = str(e)
            outputs.append(item)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    response = {"source_rag_tab_id": body.rag_tab_id, "unit_list": body.unit_list, "outputs": outputs}
    try:
        supabase = get_supabase()
        update_payload = {
            "unit_list": body.unit_list,
            "system_prompt_instruction": body.system_prompt_instruction or "",
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
            "updated_at": now_utc_iso(),
        }
        # llm_api_key 不寫入 Rag 表（該表無此欄位）；tab/quiz/create、tab/quiz/grade 依 person_id 從 User 表取得
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", body.rag_tab_id).eq("person_id", pid).execute()
    except Exception:
        pass
    return response


def _do_delete_rag_file(pid: str, fid: str):
    """共用：將 Rag 表該筆 deleted 設為 true 並刪除 storage 資料夾。"""
    try:
        supabase = get_supabase()
        supabase.table("Rag").update({"deleted": True, "updated_at": now_utc_iso()}).eq("rag_tab_id", fid).eq("person_id", pid).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Rag 表失敗: {e}")
    try:
        folder_deleted = delete_tab_folder(pid, fid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return folder_deleted


@router.post("/tab/delete/{rag_tab_id}", status_code=200)
def delete_rag_file(
    rag_tab_id: str = PathParam(..., description="要刪除的 rag_tab_id"),
    x_person_id: str | None = Header(None, alias="X-Person-Id"),
):
    """
    POST /rag/tab/delete/{rag_tab_id}，person_id 請帶 Header X-Person-Id。
    軟刪除：將 Rag 表該筆 deleted 設為 true，並刪除 storage/{person_id}/{rag_tab_id}/ 整個資料夾。
    """
    pid = (x_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 Header X-Person-Id（person_id）")
    fid = (rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    folder_deleted = _do_delete_rag_file(pid, fid)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "rag_tab_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }
