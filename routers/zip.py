"""
ZIP 與 RAG 相關 API 模組。
提供：
- GET /rag/tabs：列出 Rag 表（含 quizzes、answers）；僅回傳 query person_id 與 Rag.person_id 相同之列；query `local` 篩選 Rag.local，未傳時依連線是否本機判定；回傳依 created_at 舊→新
- GET /rag/tab/for-exam：依連線讀取 System_Setting（rag_localhost / rag_deploy）的 rag_id，回傳對應 Rag
- POST /rag/tab/create：建立一筆 Rag（可傳 local）
- PUT /rag/tab/tab-name：更新既有 Rag 的 tab_name（body：rag_id、tab_name；與 tab/create 回傳之 rag_id 相同）
- POST /rag/tab/upload-zip：上傳 ZIP
- POST /rag/tab/build-rag-zip：依 unit_list 打包並建 RAG；回應為 NDJSON 串流（start／building／unit／complete）。POST /rag/tab/build-rag-zip-stream 為同行為之別名
- POST /rag/tab/delete/{rag_tab_id}：依 rag_tab_id 軟刪除並刪除儲存（須傳 query person_id）
"""

# 引入 io 用於 BytesIO 等
import io
# 引入 json 用於 NDJSON 串流事件
import json
# 引入 os 用於 tempfile mkstemp 後 close fd
import os
# 引入 time：重試間隔
import time
# 引入 logging 用於記錄錯誤
import logging
# 引入 uuid 用於產生 repack tab_id
import uuid
# 引入 tempfile：repack 建 RAG 時寫入暫存，避免依賴上傳後立刻從 Storage 下載
import tempfile
# 引入 zipfile 用於讀取 ZIP
import zipfile

from storage3.exceptions import StorageApiError
# 引入 Path 用於路徑操作
from pathlib import Path
# 引入 Any 型別
from typing import Any

# 引入 FastAPI 的 APIRouter、HTTPException、UploadFile、File、Form、PathParam、Query、Request
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Path as PathParam, Query, Request
# 串流回應（RAG 建置進度）
from fastapi.responses import StreamingResponse

from dependencies.person_id import PersonId
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 引入台北時間工具
from utils.datetime_utils import now_taipei_iso, to_taipei_iso
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

# Rag.file_size 與 file_metadata["file_size"] 單位：二進位 MB（MiB，bytes / 1024²）
BYTES_PER_MB = 1024 * 1024


def _bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / BYTES_PER_MB


def _rag_default_row(
    rag_tab_id: str,
    *,
    tab_name: str | None = None,
    person_id: str | None = None,
    system_prompt_instruction: str | None = None,
    file_metadata: Any = None,
    local: bool = False,
) -> dict[str, Any]:
    """Rag 表一筆新增時的預設欄位；鍵順序同 public.Rag（rag_tab_id→…；不含 rag_id；created_at／updated_at 為台北時間）。"""
    ts = now_taipei_iso()
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
    row["created_at"] = ts
    row["updated_at"] = ts
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
    """GET /rag/tabs 回應：僅含 query person_id 與 Rag.person_id 相符之 Rag；每筆另含關聯的 Rag_Quiz（quizzes，每題帶一筆 answer）、以及頂層 Rag_Answer（answers）。"""
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


# 空 ZIP 檔最小約 22 bytes；低於此視為無效產物
_MIN_RAG_ZIP_BYTES = 22

# 讀取「上傳 ZIP」：遠端 metadata／下載可能延遲，重試直到成功（有上限避免無限等待）
_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS = 80
_SOURCE_UPLOAD_ZIP_SLEEP_SEC = 0.45

# RAG ZIP 上傳後讀回驗證（單次建置流程內多次嘗試）
_RAG_ZIP_VERIFY_MAX_ATTEMPTS = 18
_RAG_ZIP_VERIFY_SLEEP_INITIAL = 0.35
_RAG_ZIP_VERIFY_SLEEP_MAX = 3.0

# 單一輸出單元：整段建 RAG（含上傳與驗證）失敗時重試次數
_RAG_UNIT_FULL_BUILD_ATTEMPTS = 3
_RAG_UNIT_FULL_BUILD_RETRY_SLEEP_BASE = 0.65


def _try_read_upload_zip_once(pid: str, rag_tab_id: str) -> Path | None:
    """嘗試下載並開啟上傳 ZIP；成功回傳暫存 Path（呼叫端負責 unlink），失敗回傳 None。"""
    path = get_zip_path(rag_tab_id) or get_zip_path_by_person(pid, rag_tab_id)
    if not path or not path.exists():
        return None
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.namelist()
        return path
    except zipfile.BadZipFile:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _fetch_source_upload_zip_with_retries(pid: str, rag_tab_id: str) -> Path:
    """
    直到能從儲存讀回並以 ZipFile 開啟上傳 ZIP，或超過重試上限。
    回傳本機暫存路徑（與原本 get_zip_path 相同語意，由 finally 刪除）。
    """
    last_fail = "無法取得路徑或下載失敗"
    for attempt in range(_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS):
        path = _try_read_upload_zip_once(pid, rag_tab_id)
        if path is not None:
            return path
        last_fail = f"第 {attempt + 1}/{_SOURCE_UPLOAD_ZIP_MAX_ATTEMPTS} 次仍無法讀取上傳 ZIP"
        time.sleep(_SOURCE_UPLOAD_ZIP_SLEEP_SEC)
    raise HTTPException(
        status_code=503,
        detail=(
            f"多次重試後仍無法讀取上傳的 ZIP（rag_tab_id={rag_tab_id}）。"
            f"若剛上傳完請稍後再試；{last_fail}"
        ),
    )


def _verify_saved_rag_zip_readable(rag_zip_tab_id: str) -> str | None:
    """
    確認 RAG ZIP 已寫入儲存且可下載為非空檔案；失敗時間隔重試（應對 metadata／下載延遲）。
    成功回傳 None；全數失敗回傳最後一則錯誤說明。
    """
    delay = _RAG_ZIP_VERIFY_SLEEP_INITIAL
    last_err = "RAG ZIP 上傳後無法從儲存讀回驗證"
    for _ in range(_RAG_ZIP_VERIFY_MAX_ATTEMPTS):
        verify_path = get_zip_path(rag_zip_tab_id)
        if not verify_path or not verify_path.exists():
            last_err = "RAG ZIP 上傳後無法從儲存讀回驗證"
            time.sleep(delay)
            delay = min(delay * 1.3, _RAG_ZIP_VERIFY_SLEEP_MAX)
            continue
        try:
            sz = verify_path.stat().st_size
            if sz < _MIN_RAG_ZIP_BYTES:
                last_err = "RAG ZIP 驗證失敗（讀回檔案過小或損毀）"
            else:
                return None
        finally:
            try:
                verify_path.unlink(missing_ok=True)
            except Exception:
                pass
        time.sleep(delay)
        delay = min(delay * 1.3, _RAG_ZIP_VERIFY_SLEEP_MAX)
    return last_err


def _build_one_rag_zip_output_item(
    body: PackRequest,
    pid: str,
    api_key: str,
    zip_bytes: bytes,
    filename: str,
) -> dict[str, Any]:
    """
    將單一 repack 單元上傳至 repack、建 RAG ZIP 上傳至 rag；建庫使用記憶體 repack bytes（暫存檔）。
    RAG 上傳後會多次重試讀回驗證；若該單元仍失敗則整段流程最多重試 _RAG_UNIT_FULL_BUILD_ATTEMPTS 次。
    回傳與 build-rag-zip outputs[] 單筆相同結構；失敗時含 rag_error（最終失敗會附「已重試 N 次」）。
    """
    repack_tab_id = Path(filename).stem if filename else None
    if not repack_tab_id or "/" in repack_tab_id or "\\" in repack_tab_id:
        repack_tab_id = str(uuid.uuid4())

    last_item: dict[str, Any] = {}
    for attempt in range(_RAG_UNIT_FULL_BUILD_ATTEMPTS):
        item: dict[str, Any] = {
            "filename": filename,
            "unit_name": repack_tab_id,
            "repack_filename": f"{repack_tab_id}.zip",
            "rag_filename": f"{repack_tab_id}_rag.zip",
        }
        rag_bytes_out: bytes | None = None
        try:
            from utils.rag_faiss_zip import make_rag_zip_from_zip_path

            tab_id = save_zip(
                zip_bytes,
                filename,
                folder=FOLDER_REPACK,
                person_id=pid,
                parent_tab_id=body.rag_tab_id,
                tab_id=repack_tab_id,
            )
            rag_zip_tab_id = f"{tab_id}_rag"
            item["unit_name"] = tab_id
            item["repack_filename"] = f"{tab_id}.zip"
            item["rag_filename"] = f"{rag_zip_tab_id}.zip"

            fd, repack_tmp = tempfile.mkstemp(suffix=".zip", prefix="myquizai_repack_")
            os.close(fd)
            repack_local = Path(repack_tmp)
            try:
                repack_local.write_bytes(zip_bytes)
                rag_bytes_out = make_rag_zip_from_zip_path(
                    repack_local,
                    api_key,
                    chunk_size=body.chunk_size,
                    chunk_overlap=body.chunk_overlap,
                )
            finally:
                try:
                    repack_local.unlink(missing_ok=True)
                except Exception:
                    pass

            if not rag_bytes_out or len(rag_bytes_out) < _MIN_RAG_ZIP_BYTES:
                item["rag_error"] = "RAG ZIP 產物無效或過小（未產生可用向量庫 ZIP）"
            else:
                save_zip(
                    rag_bytes_out,
                    f"{tab_id}.zip",
                    folder=FOLDER_RAG,
                    person_id=pid,
                    parent_tab_id=body.rag_tab_id,
                    tab_id=rag_zip_tab_id,
                )
                verify_err = _verify_saved_rag_zip_readable(rag_zip_tab_id)
                if verify_err:
                    item["rag_error"] = verify_err
        except ValueError as e:
            item["rag_error"] = str(e)
        except Exception as e:
            item["rag_error"] = str(e)

        item["file_size"] = _bytes_to_mb(len(rag_bytes_out) if rag_bytes_out is not None else len(zip_bytes))

        if not item.get("rag_error"):
            return item
        last_item = item
        if attempt < _RAG_UNIT_FULL_BUILD_ATTEMPTS - 1:
            time.sleep(_RAG_UNIT_FULL_BUILD_RETRY_SLEEP_BASE * (attempt + 1))

    err = (last_item.get("rag_error") or "未知錯誤").strip()
    last_item["rag_error"] = f"{err}（已重試 {_RAG_UNIT_FULL_BUILD_ATTEMPTS} 次）"
    return last_item


def _rag_zip_build_counts(outputs: list[dict[str, Any]]) -> dict[str, int]:
    """供 build-rag-zip 串流 complete 事件使用：總筆數、成功／失敗筆數。"""
    total = len(outputs)
    failed = sum(1 for o in outputs if o.get("rag_error"))
    return {"total": total, "built_ok": total - failed, "built_failed": failed}


def _persist_rag_build_metadata(body: PackRequest, pid: str, response: dict[str, Any]) -> None:
    """成功建置後寫入 Rag 表（失敗時靜默略過，與 build_rag_zip 成功路徑一致）。"""
    try:
        supabase = get_supabase()
        update_payload = {
            "unit_list": body.unit_list,
            "system_prompt_instruction": body.system_prompt_instruction or "",
            "rag_metadata": response,
            "chunk_size": body.chunk_size,
            "chunk_overlap": body.chunk_overlap,
            "updated_at": now_taipei_iso(),
        }
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", body.rag_tab_id).eq("person_id", pid).execute()
    except Exception:
        pass


@router.get("/tabs", response_model=ListRagResponse)
def list_rag(
    request: Request,
    person_id: PersonId,
    local: bool | None = Query(
        None,
        description="僅回傳 Rag.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false（與前端在本機開發傳 true、正式環境傳 false 一致）",
    ),
):
    """
    列出 Rag 表內容，僅回傳 deleted=False 的資料，且 Rag.local 須與 query `local` 相符（未傳 `local` 時依連線是否本機自動判定）。
    僅回傳 query 所傳 person_id 與 Rag.person_id 相同之列（與 GET /exam/tabs 一致）。
    回傳列依 created_at 由舊到新排序。
    每筆 Rag 含表上所有欄位，並帶關聯的 Rag_Quiz（quizzes）與 Rag_Answer（answers）。
    關聯方式：quizzes 下每筆 quiz 帶 answers（依 rag_quiz_id 關聯，每題僅一筆）；頂層 answers 為該 rag 下全部答案的扁平列表。
    LLM API Key 依 person_id 從 User 表取得。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _rag_table_select(RAG_SELECT_ALL, exclude_deleted=True, local_match=local_filter)
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]
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
def get_for_exam_rag(request: Request, _person_id: PersonId):
    """
    依連線讀取 System_Setting（本機：rag_localhost，否則 rag_deploy）的 value 作為 rag_id，取得該筆 Rag（deleted=false）。
    回傳格式與 POST /rag/tab/build-rag-zip 一致，並多帶 rag_id、rag_tab_id、llm_api_key、system_prompt_instruction、file_size、file_metadata（與 Rag 表欄位一致；file_size 為上傳 ZIP 之 MB）。
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
                "file_size": None,
                "file_metadata": None,
            }
        row = data[0]
        meta = row.get("rag_metadata")
        extra = {
            "rag_id": row.get("rag_id"),
            "rag_tab_id": row.get("rag_tab_id"),
            "llm_api_key": row.get("llm_api_key"),
            "system_prompt_instruction": row.get("system_prompt_instruction"),
            "file_size": row.get("file_size"),
            "file_metadata": row.get("file_metadata"),
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
def create_unit(body: CreateRagRequest, caller_person_id: PersonId):
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
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
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
            "created_at": to_taipei_iso(row.get("created_at")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tab/tab-name")
def update_unit_tab_name(body: UpdateRagUnitNameRequest, caller_person_id: PersonId):
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
        if ((pid or "").strip() != caller_person_id):
            raise HTTPException(status_code=403, detail="無權修改該 Rag")
        ts = now_taipei_iso()
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
    caller_person_id: PersonId,
    file: UploadFile = File(...),
    rag_tab_id: str = Form(..., description="對應 tab/create 建立的 rag_tab_id（Rag 表 rag_tab_id），ZIP 會存於此路徑"),
    person_id: str = Form(..., description="寫入儲存路徑的 person_id，需與 tab/create 一致"),
):
    """
    Upload Zip：只做上傳並寫入資料庫。需先以 tab/create 建立該 rag_tab_id 的 Rag 資料。
    傳入 rag_tab_id（tab/create 的 rag_tab_id）、ZIP 檔案與 person_id（Form 必填）。
    出題/評分時由後端依 person_id 從 User 表取得 LLM API Key。
    會更新該筆 Rag 的 file_metadata（filename、second_folders、file_size 等）與 file_size 欄位（皆為 MB）。
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
    if resolved_person_id != caller_person_id:
        raise HTTPException(status_code=400, detail="Form 的 person_id 與 query 不一致")

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
    except StorageApiError as e:
        if e.status == 413:
            raise HTTPException(
                status_code=413,
                detail=(
                    "ZIP 超過 Supabase Storage 允許的單檔大小上限；請縮小檔案，"
                    "或至 Supabase Dashboard → Project Settings → Storage 調高／確認方案限制。"
                ),
            ) from e
        raise HTTPException(status_code=502, detail=f"儲存上傳失敗: {e.message}") from e

    file_size_mb = _bytes_to_mb(len(contents))
    file_metadata = {
        "rag_id": row["rag_id"],
        "rag_tab_id": fid,
        "created_at": to_taipei_iso(row["created_at"]),
        "filename": file.filename,
        "second_folders": folders,
        "file_size": file_size_mb,
    }
    update_payload: dict[str, Any] = {
        "file_metadata": file_metadata,
        "file_size": file_size_mb,
        "updated_at": now_taipei_iso(),
    }
    # llm_api_key 不寫入 Rag 表（該表無此欄位）；依 person_id 從 User 表取得
    try:
        supabase.table("Rag").update(update_payload).eq("rag_tab_id", fid).eq("person_id", resolved_person_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return file_metadata


@router.post("/tab/build-rag-zip-stream", include_in_schema=False)
@router.post("/tab/build-rag-zip")
def build_rag_zip(body: PackRequest, caller_person_id: PersonId):
    """
    依先前上傳的 ZIP（rag_tab_id）與 unit_list 打包並建 RAG（FAISS）ZIP；LLM API Key 依 person_id 從 User 表取得。
    **回應為 NDJSON 串流**（`application/x-ndjson`），請以 `fetch` 讀取 `response.body`，勿使用單次 `response.json()`。
    每一輸出單元須：**產出有效 RAG ZIP bytes**、**成功上傳至 rag 儲存**，且**上傳後能自儲存讀回非空檔**（讀回會多次重試）。
    單一單元整段建置若仍失敗，會**自動再試最多 3 次**後才寫入 `rag_error`。讀取一開始的「上傳 ZIP」亦會**反覆重試**直到可開啟或達上限（逾限回 503）。
    整批任一有 `rag_error` 則 `complete.success` 為 false（不寫入 Rag 表）。

    事件列舉（每行一個物件）：
    - `{"type":"start","total":N,"source_rag_tab_id":"...","unit_list":"..."}`：即將處理 N 個輸出單元
    - `{"type":"building","index":i,"total":N,"completed_before":i-1,"filename":"..."}`：即將開始建第 i 個 RAG ZIP（`completed_before` 為已跑完筆數；`filename` 為 repack 檔名）
    - `{"type":"unit","index":i,"total":N,"output":{...}}`：第 i 個單元完成（`output` 含 `unit_name`、`filename`（repack 工作檔名）、`repack_filename`／`rag_filename`（bucket 內 `{tab_id}.zip`／`{tab_id}_rag.zip`）、`file_size`，可含 `rag_error`）
    - `{"type":"complete","success":bool,"total","built_ok","built_failed","source_rag_tab_id","unit_list","outputs"}`：全部結束；`success` 為 false 時另有 `message` 說明，且**不會**更新 Rag 表（與原 API 整批失敗行為一致）

    串流階段 HTTP 狀態碼固定 **200**（需先送出標頭）；請以最後一則 `type===complete` 的 `success` 判斷整批成敗；成功時寫入 Rag 表 rag_metadata。
    多次重試後仍無法讀取上傳 ZIP 時回 **503**。其餘驗證失敗（BadZip、無 API Key 等）仍回 **400** 與一般 JSON `detail`。

    `POST /rag/tab/build-rag-zip-stream` 與本端點相同，僅自 OpenAPI 隱藏，供舊客戶端相容。
    """
    pid = (body.person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")

    path = _fetch_source_upload_zip_with_retries(pid, body.rag_tab_id)

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.unit_list)
    if not packed:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="unit_list 為空或格式錯誤，例：220222+220301")

    api_key = get_llm_api_key_for_person(pid)
    if not api_key:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail="該使用者（person_id）尚未於個人設定填寫 LLM API Key，請至 User 設定",
        )

    total = len(packed)

    def ndjson_events():
        outputs: list[dict[str, Any]] = []
        try:
            yield (
                json.dumps(
                    {
                        "type": "start",
                        "total": total,
                        "source_rag_tab_id": body.rag_tab_id,
                        "unit_list": body.unit_list,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for idx, (zip_bytes, filename) in enumerate(packed):
                yield (
                    json.dumps(
                        {
                            "type": "building",
                            "index": idx + 1,
                            "total": total,
                            "completed_before": idx,
                            "filename": filename,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                item = _build_one_rag_zip_output_item(body, pid, api_key, zip_bytes, filename)
                outputs.append(item)
                yield (
                    json.dumps(
                        {"type": "unit", "index": idx + 1, "total": total, "output": item},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            success = not any(o.get("rag_error") for o in outputs)
            counts = _rag_zip_build_counts(outputs)
            response = {
                "source_rag_tab_id": body.rag_tab_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if success:
                _persist_rag_build_metadata(body, pid, response)
            complete_ev: dict[str, Any] = {
                "type": "complete",
                "success": success,
                "source_rag_tab_id": body.rag_tab_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if not success:
                complete_ev["message"] = "RAG ZIP 建立失敗（請修正後重試）"
            yield json.dumps(complete_ev, ensure_ascii=False) + "\n"
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(
        ndjson_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


def _do_delete_rag_file_by_tab_id(fid: str) -> tuple[bool, str]:
    """依 rag_tab_id 將 Rag 未刪除列軟刪除，並對各 person_id 刪除 Supabase storage 資料夾。回傳 (folder_deleted, 回傳用 person_id)。"""
    supabase = get_supabase()
    sel = supabase.table("Rag").select("person_id").eq("rag_tab_id", fid).eq("deleted", False).execute()
    if not sel.data:
        raise HTTPException(status_code=404, detail="找不到該 rag_tab_id 的 Rag 資料，或已刪除")
    pids_ordered: list[str] = []
    seen: set[str] = set()
    for row in sel.data:
        pid = (row.get("person_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        pids_ordered.append(pid)
    primary_pid = pids_ordered[0] if pids_ordered else ""
    try:
        supabase.table("Rag").update({"deleted": True, "updated_at": now_taipei_iso()}).eq("rag_tab_id", fid).eq("deleted", False).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新 Rag 表失敗: {e}")
    folder_deleted = False
    for pid in pids_ordered:
        try:
            if delete_tab_folder(pid, fid):
                folder_deleted = True
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return folder_deleted, primary_pid


@router.post("/tab/delete/{rag_tab_id}", status_code=200)
def delete_rag_file(
    _person_id: PersonId,
    rag_tab_id: str = PathParam(..., description="要刪除的 rag_tab_id"),
):
    """
    POST /rag/tab/delete/{rag_tab_id}。
    軟刪除：將 Rag 表該 rag_tab_id 之未刪除列 deleted 設為 true，並刪除各 person_id 下 storage/{person_id}/{rag_tab_id}/ 對應之資料夾。
    """
    fid = (rag_tab_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_tab_id")
    folder_deleted, pid = _do_delete_rag_file_by_tab_id(fid)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "rag_tab_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }
