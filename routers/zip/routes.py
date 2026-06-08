"""routers.zip routes（自 zip.py 拆分）。"""

import base64
import logging
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from postgrest.exceptions import APIError
from fastapi.responses import StreamingResponse

from dependencies.person_id import PersonId
from dependencies.course_id import CourseId

from utils.llm_key import get_rag_api_key
from utils.openapi import openapi_body
from utils.serialization import to_json_safe
from utils.zip_utils import (
    build_folder_map,
    repack_tasks_to_zips,
)
from utils.taipei_time import now_taipei_iso
from utils.supabase import get_supabase
from utils.rag_course import (
    execute_with_course_id_fallback,
    insert_rag_child_row,
    require_rag_tab_owner,
    resolve_rag_tab_owner_person_id,
    select_without_course_id_if_needed,
)
from utils.rag_stem import transcript_from_row
from utils.rag_exam_setting import is_localhost_request
from utils.media import audio_media_type_for_suffix
from utils.rag_transcript import (
    pick_audio_from_upload_zip,
    pick_audio_from_upload_zip_with_folder_fallback,
    read_mp3_unit_transcript_from_upload_zip,
    read_repack_zip_bytes,
    read_single_transcript_text_from_upload_zip,
    read_supplementary_text_from_youtube_unit,
    read_upload_zip_bytes,
    read_youtube_video_id_from_upload_zip,
)


from utils.fs import safe_unlink
from .schemas import (
    InsertRagQuizRowRequest,
    ListRagResponse,
    PackRequest,
    RagUnitMp3FilePreviewResponse,
    RagUnitMp3FileResponse,
    RagUnitTextPreviewResponse,
    RagUnitTextResponse,
    RagUnitYoutubeUrlPreviewResponse,
    RagUnitYoutubeUrlResponse,
    UpdateRagQuizQuizNameRequest,
    UpdateRagUnitNameRequest,
)
from .helpers import (
    RAG_UNIT_TYPE_MP3,
    RAG_UNIT_TYPE_RAG,
    RAG_UNIT_TYPE_TEXT,
    RAG_UNIT_TYPE_YOUTUBE,
    _build_one_rag_zip_output_item,
    _chunk_params_per_task,
    _create_rag_record,
    _do_delete_rag_file_by_page_id,
    _fetch_source_upload_zip_with_retries,
    _fetch_user_type,
    _ndjson_line,
    _persist_rag_build_metadata,
    _quizzes_by_rag_unit_ids,
    _rag_table_select,
    _rag_zip_build_counts,
    _unit_name_overrides_per_task,
    _unit_types_per_task,
    _units_by_rag_page_ids,
    _upload_rag_zip_contents,
    _validate_rag_tab_create_fields,
)

_logger = logging.getLogger("routers.zip")

router = APIRouter(prefix="/rag", tags=["rag"])

RAG_SELECT_ALL = "*"


@router.get("/pages", response_model=ListRagResponse)
def list_rag(
    request: Request,
    person_id: PersonId,
    course_id: CourseId,
    local: bool | None = Query(
        None,
        description="僅回傳 Rag.local 與此值相同的列。未傳時：連線來源為 127.0.0.1、localhost、::1 視為 true，否則 false",
    ),
):
    """
    列出 Rag 表內容（deleted=False），須傳 course_id，僅回傳該課程的 Rag／Rag_Unit／Rag_Quiz；
    且僅回傳與 query person_id 相符之列，Rag.local 須與 query local 相符（未傳 local 時依連線自動判定）。
    回傳列依 created_at 由舊到新排序。
    每筆 Rag 含 units（Rag_Unit 列表），每個 unit 含 quizzes（Rag_Quiz 列表，含 follow_up、quiz_history_list）。
    音訊單元（unit_type=3）且 mp3_file_name 非空時，另含 mp3_audio_url：相對於 API 根路徑的 GET /rag/units/{rag_unit_id}/mp3-file 查詢字串（`rag_page_id`，不需 person_id），可接在後端 origin 後作為 `<audio src>`。
    YouTube 單元（unit_type=4）且 youtube_url 非空時，另含 youtube_url_api：相對於 API 根路徑的 GET /rag/units/{rag_unit_id}/youtube-url 查詢字串（`rag_page_id`，不需 person_id）。
    """
    try:
        local_filter = local if local is not None else is_localhost_request(request)
        data = _rag_table_select(
            RAG_SELECT_ALL,
            exclude_deleted=True,
            local_match=local_filter,
            course_id=course_id,
        )
        pid = person_id.strip()
        data = [r for r in data if (r.get("person_id") or "").strip() == pid]

        rag_page_ids = list(dict.fromkeys(
            r.get("rag_page_id") for r in data if r.get("rag_page_id")
        ))
        units_by_tab = _units_by_rag_page_ids(rag_page_ids, course_id=course_id)

        all_unit_ids: list[int] = []
        for units in units_by_tab.values():
            for unit in units:
                uid = unit.get("rag_unit_id")
                if uid is not None:
                    try:
                        all_unit_ids.append(int(uid))
                    except (TypeError, ValueError):
                        pass
        all_unit_ids = list(dict.fromkeys(all_unit_ids))
        quizzes_by_unit = _quizzes_by_rag_unit_ids(all_unit_ids, course_id=course_id)

        for row in data:
            page_id = row.get("rag_page_id")
            units = units_by_tab.get(page_id, []) if page_id else []
            for unit in units:
                uid = unit.get("rag_unit_id")
                uid_int = int(uid) if uid is not None else None
                unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []
                try:
                    utype = int(unit.get("unit_type") or 0)
                except (TypeError, ValueError):
                    utype = 0
                unit_name_q = (unit.get("unit_name") or "").strip()
                folder_c = (unit.get("folder_combination") or "").strip()
                if (
                    utype == RAG_UNIT_TYPE_MP3
                    and (unit.get("mp3_file_name") or "").strip()
                    and (unit_name_q or folder_c)
                    and page_id
                    and uid_int is not None
                ):
                    unit["mp3_audio_url"] = (
                        f"/rag/units/{uid_int}/mp3-file?"
                        + urlencode(
                            {
                                "rag_page_id": str(page_id).strip(),
                                "course_id": str(course_id),
                            }
                        )
                    )
                if (
                    utype == RAG_UNIT_TYPE_YOUTUBE
                    and (unit.get("youtube_url") or "").strip()
                    and page_id
                    and uid_int is not None
                ):
                    unit["youtube_url_api"] = (
                        f"/rag/units/{uid_int}/youtube-url?"
                        + urlencode(
                            {
                                "rag_page_id": str(page_id).strip(),
                                "course_id": str(course_id),
                            }
                        )
                    )
            row["units"] = units

        data = to_json_safe(data)
        return ListRagResponse(rags=data, count=len(data))
    except Exception as e:
        _logger.exception("GET /rag/pages 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag 失敗: {e!s}")


@router.patch("/pages/{rag_page_id}")
def update_unit_tab_name(
    body: openapi_body(UpdateRagUnitNameRequest, {"tab_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="要更新 tab_name 的 rag_page_id"),
):
    """
    更新既有 Rag 的 tab_name。以 rag_page_id 比對；僅更新 deleted=false 的列。
    回傳 rag_id、rag_page_id、person_id、tab_name、updated_at。
    """
    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")
    tab_name = (body.tab_name or "").strip()
    if not tab_name:
        raise HTTPException(status_code=400, detail="請傳入 tab_name")
    try:
        supabase = get_supabase()
        sel = (
            supabase.table("Rag")
            .select("rag_id, rag_page_id, person_id, course_id")
            .eq("rag_page_id", fid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not sel.data or len(sel.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_page_id 的 Rag 資料，或已刪除")
        row = sel.data[0]
        rag_id = row.get("rag_id")
        pid = row.get("person_id")
        if ((pid or "").strip() != caller_person_id):
            raise HTTPException(status_code=403, detail="無權修改該 Rag")
        ts = now_taipei_iso()
        supabase.table("Rag").update({"tab_name": tab_name, "updated_at": ts}).eq("rag_page_id", fid).eq("course_id", course_id).eq("deleted", False).execute()
        return {
            "rag_id": rag_id,
            "rag_page_id": fid,
            "person_id": pid,
            "tab_name": tab_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pages/{rag_page_id}", status_code=200, summary="Delete Rag File", operation_id="rag_tab_delete")
def delete_rag_file(
    _person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="要刪除的 rag_page_id"),
):
    """
    DELETE /rag/pages/{rag_page_id}。
    軟刪除：將 Rag 表該 rag_page_id 之未刪除列 deleted 設為 true，同時軟刪除所有對應 Rag_Unit，並刪除 storage 資料夾。
    """
    fid = (rag_page_id or "").strip()
    if not fid or "/" in fid or "\\" in fid:
        raise HTTPException(status_code=400, detail="無效的 rag_page_id")
    folder_deleted, pid = _do_delete_rag_file_by_page_id(fid, course_id)
    return {
        "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
        "rag_page_id": fid,
        "person_id": pid,
        "rag_updated": True,
        "folder_deleted": folder_deleted,
    }


@router.post("/pages/upload-zip", status_code=201)
async def create_upload_zip(
    caller_person_id: PersonId,
    course_id: CourseId,
    file: UploadFile = File(...),
    rag_page_id: str = Form(..., description="Rag 的 tab 識別，對應 Rag 表 rag_page_id 欄位"),
    person_id: str | None = Form(None, description="選填；未傳以 token 解析的呼叫者為準；有傳須與呼叫者一致"),
    tab_name: str = Form(..., description="Rag 顯示名稱，寫入 Rag 表 tab_name 欄位"),
    local: bool = Form(False, description="是否為本機 RAG，寫入 Rag 表 local 欄位"),
):
    """
    建立 Rag 並上傳 ZIP。
    multipart/form-data：file、rag_page_id、tab_name、local（選填，預設 false）；person_id 選填（未傳以 token 呼叫者為準）。
    須傳 query course_id。
    回傳 create 欄位與 file_metadata。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="請上傳 .zip 檔案")

    fid, pid, name = _validate_rag_tab_create_fields(
        rag_page_id=rag_page_id,
        person_id=person_id,
        tab_name=tab_name,
        caller_person_id=caller_person_id,
    )

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取上傳檔案")

    try:
        create_result = _create_rag_record(
            rag_page_id=fid,
            person_id=pid,
            tab_name=name,
            course_id=course_id,
            local=local,
        )
        file_metadata = _upload_rag_zip_contents(
            contents=contents,
            filename=file.filename,
            rag_page_id=fid,
            person_id=pid,
            course_id=course_id,
        )
        return {**create_result, "file_metadata": file_metadata}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pages/{rag_page_id}/build-zip-stream", include_in_schema=False)
@router.post("/pages/{rag_page_id}/build-zip")
def build_rag_zip(
    body: openapi_body(
        PackRequest,
        {
            "unit_list": "folder1",
            "unit_names": "",
            "unit_types": "",
            "transcripts": None,
            "rag_chunk_size": 1000,
            "rag_chunk_overlap": 200,
            "rag_chunk_sizes": "",
            "rag_chunk_overlaps": "",
            "build_faiss": None,
        },
    ),
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="來源上傳 ZIP 的 rag_page_id"),
    repack_only: bool = Query(
        False,
        description="為 True 時強制不建 FAISS（unit_type=1 時 rag 改為 repack 複製）；不影響 unit_type=2/3/4 之逐字稿 rag ZIP",
    ),
):
    """
    依先前上傳的 ZIP（path 參數 rag_page_id）與 unit_list 重新打包。
    **FAISS 建置規則（逐 unit 判斷）**：`user_type==1`（且未強制關閉）且該 unit 之 `unit_type==1 (rag)` → 建 FAISS 並上傳至 rag；`unit_type` 為 2／3／4 時仍 repack 原 ZIP，但 **rag 區上傳內含單一 `transcript.md`（逐字稿全文）之 ZIP**，非 repack 複製；其餘 unit_type==0 等 → repack 同內容複製至 rag。
    可選 query **repack_only=true**：強制全部 unit 不建 FAISS；**不影響** 2／3／4 之逐字稿 rag ZIP 行為。
    可選 body **build_faiss**：`false` 同 repack_only；`true` 強制允許 FAISS（仍需 unit_type==1 觸發）；省略時依 user_type 判定。
    LLM API Key 僅在「最終會建 FAISS」（do_rag 為 True）時必填（依 course_id 自 Course_Setting key=rag-api-key 取得；見 PUT /v1/rag/llm-api-key）。
    body.unit_types 為選填，與 unit_list 逗號分段對齊；**未傳或該段為 0** 時會依單元 ZIP 推斷（恰一音訊＋一文字檔→3、僅一個 .md 等→2；**YouTube 仍須明確傳 4**）。寫入各 Rag_Unit.unit_type。**推斷為 2** 且來源為 `.md`/`.txt` 時 **Rag_Unit.transcript** 為檔案 UTF-8 全文（含 Markdown）。
    body.transcripts 為選填，與 unit_list 逗號分段同序；索引 i 之字串若非空白，覆寫該單元逐字稿（Markdown UTF-8 原樣），仍自 ZIP 擷取 text_file_name／mp3_file_name／youtube_url。
    body.unit_names 為選填，與 packed 任務同序（逗號字串或 JSON 字串陣列）；該段非空白時覆寫串流 output.unit_name 與寫入之 Rag_Unit.unit_name（顯示名）。output.folder_combination 恒為 repack ZIP 檔名 stem（寫入 Rag_Unit.folder_combination；多資料夾為 ``a/tb/tc``）。

    **回應為 NDJSON 串流**（`application/x-ndjson`），請以 `fetch` 讀取 `response.body`，勿使用單次 `response.json()`。
    每一輸出單元須 **成功上傳 repack**；rag 資料夾須 **成功寫入**（unit_type=1 且建 FAISS 為向量庫 ZIP；2／3／4 為逐字稿 md ZIP；其餘為 repack 同內容），且**上傳後能自儲存讀回非空檔**。
    整批成功時自動在 Rag_Unit 表建立對應記錄（每個輸出單元一筆）並更新 Rag.rag_metadata。
    整批任一有 `rag_error` 則 `complete.success` 為 false（不寫入 Rag 表，不建立 Rag_Unit）。

    事件列舉（每行一個物件）：
    - `{"type":"start","total":N,"source_rag_page_id":"...","unit_list":"...","user_type":int,"build_faiss_request":bool|null,"repack_only":bool,"allow_faiss":bool}`（allow_faiss=各 unit 是否可建 FAISS，仍需 unit_type==1 才實際建）
    - `{"type":"building","index":i,"total":N,"completed_before":i-1,"filename":"..."}`
    - `{"type":"unit",...,"output":{...}}`：output 含 **folder_combination**（單元 repack ZIP 檔名 stem，寫入 Rag_Unit.folder_combination；多資料夾為 ``folder1/tfolder2``）、**unit_name**（顯示名，可經 unit_names 覆寫）、rag_mode（`faiss`＝向量庫；`transcript_md`＝逐字稿 md ZIP；`repack_copy`＝與 repack 同內容複製）、`transcript_plain`（鍵名沿用舊版；**unit_type=2 且來源為 .md/.txt 時為檔案 UTF-8 全文，Markdown 原樣**，與寫入 Rag_Unit.transcript 一致）；**text_file_name** 僅 **unit_type=2** 有值（來源文字檔檔名）；**mp3_file_name** 僅 3；**youtube_url** 僅 4；**rag_chunk_size**、**rag_chunk_overlap**（本任務實際使用，與 Rag_Unit 一致）；rag_filename（物件鍵仍為 *_rag.zip）
    - `{"type":"complete","success":bool,"total","built_ok","built_failed","source_rag_page_id","unit_list","outputs"}`

    串流階段 HTTP 狀態碼固定 **200**；請以最後一則 `type===complete` 的 `success` 判斷整批成敗。
    `POST /rag/pages/{rag_page_id}/build-zip-stream` 與本端點相同，僅自 OpenAPI 隱藏，供舊客戶端相容。
    """
    body.rag_page_id = (rag_page_id or "").strip()
    # person_id 選填：未傳以 token 呼叫者為準；有傳須一致（過渡期相容檢查）
    pid = (body.person_id or "").strip() or caller_person_id
    if pid != caller_person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與呼叫者（token）不一致")

    require_rag_tab_owner(pid, body.rag_page_id, course_id)

    path = _fetch_source_upload_zip_with_retries(pid, body.rag_page_id)

    try:
        with zipfile.ZipFile(path, "r") as z:
            folder_map = build_folder_map(z)
    except zipfile.BadZipFile:
        safe_unlink(path)
        raise HTTPException(status_code=400, detail="無法讀取該 ZIP 檔案")

    packed = repack_tasks_to_zips(path, folder_map, body.unit_list)
    if not packed:
        safe_unlink(path)
        raise HTTPException(status_code=400, detail="unit_list 為空或格式錯誤，例：220222+220301")

    api_key = get_rag_api_key(course_id)
    user_type_val = _fetch_user_type(pid, course_id)

    # 允許 FAISS：user_type==1 且未強制關閉；即使允許，unit_type==1 才真正觸發建置
    if repack_only or body.build_faiss is False:
        allow_faiss = False
    elif body.build_faiss is True:
        allow_faiss = True
    else:
        allow_faiss = (user_type_val == 1)

    if allow_faiss and not api_key:
        safe_unlink(path)
        raise HTTPException(
            status_code=400,
            detail="請設定 RAG API Key：PUT /v1/rag/llm-api-key（Course_Setting key=rag-api-key，依 course_id）",
        )

    total = len(packed)
    unit_types_per_task = _unit_types_per_task(body.unit_types, total)
    chunk_pairs = _chunk_params_per_task(
        body.rag_chunk_sizes,
        body.rag_chunk_overlaps,
        total,
        body.rag_chunk_size,
        body.rag_chunk_overlap,
    )
    unit_name_overrides = _unit_name_overrides_per_task(body.unit_names, total)

    def _do_rag_for_unit(ut: int) -> bool:
        """只有 allow_faiss 且 unit_type==1 (rag) 時才建 FAISS；其餘走 repack 分支（2/3/4 時 rag 為逐字稿 ZIP）。"""
        return allow_faiss and (ut == RAG_UNIT_TYPE_RAG)

    def ndjson_events():
        outputs: list[dict[str, Any]] = []
        try:
            yield _ndjson_line(
                {
                    "type": "start",
                    "total": total,
                    "source_rag_page_id": body.rag_page_id,
                    "unit_list": body.unit_list,
                    "user_type": user_type_val,
                    "build_faiss_request": body.build_faiss,
                    "repack_only": repack_only,
                    "allow_faiss": allow_faiss,
                }
            )
            for idx, (zip_bytes, filename) in enumerate(packed):
                yield _ndjson_line(
                    {
                        "type": "building",
                        "index": idx + 1,
                        "total": total,
                        "completed_before": idx,
                        "filename": filename,
                    }
                )
                ut = unit_types_per_task[idx]
                t_cs, t_co = chunk_pairs[idx]
                ov_list = body.transcripts or []
                transcript_override = ov_list[idx] if idx < len(ov_list) else None
                item = _build_one_rag_zip_output_item(
                    body,
                    pid,
                    api_key or "",
                    zip_bytes,
                    filename,
                    do_rag=_do_rag_for_unit(ut),
                    unit_type=ut,
                    task_rag_chunk_size=t_cs,
                    task_rag_chunk_overlap=t_co,
                    transcript_override=transcript_override,
                )
                name_ov = unit_name_overrides[idx] if idx < len(unit_name_overrides) else None
                if name_ov is not None:
                    item["unit_name"] = name_ov
                try:
                    ut_out = int(item.get("unit_type") or 0)
                except (TypeError, ValueError):
                    ut_out = 0
                if ut_out != RAG_UNIT_TYPE_RAG:
                    item["rag_chunk_size"] = 0
                    item["rag_chunk_overlap"] = 0
                outputs.append(item)
                yield _ndjson_line(
                    {"type": "unit", "index": idx + 1, "total": total, "output": item}
                )
            success = not any(o.get("rag_error") for o in outputs)
            counts = _rag_zip_build_counts(outputs)
            response = {
                "source_rag_page_id": body.rag_page_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if success:
                _persist_rag_build_metadata(body, pid, course_id, response)
            complete_ev: dict[str, Any] = {
                "type": "complete",
                "success": success,
                "source_rag_page_id": body.rag_page_id,
                "unit_list": body.unit_list,
                "outputs": outputs,
                **counts,
            }
            if not success:
                complete_ev["message"] = "RAG ZIP 建立失敗（請修正後重試）"
            yield _ndjson_line(complete_ev)
        finally:
            safe_unlink(path)

    return StreamingResponse(
        ndjson_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/pages/{rag_page_id}/units")
def list_rag_units(
    _caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="要列出 Rag_Unit 的 rag_page_id"),
):
    """
    依 rag_page_id 列出所有未刪除的 Rag_Unit，每個 unit 含關聯的 Rag_Quiz（quizzes，含 follow_up）。
    依 created_at 由舊到新排序。
    """
    try:
        fid = (rag_page_id or "").strip()
        if not fid:
            raise HTTPException(status_code=400, detail="請傳入 rag_page_id")
        supabase = get_supabase()

        def build_units_query(with_course_filter: bool):
            q = (
                supabase.table("Rag_Unit")
                .select("*")
                .eq("rag_page_id", fid)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.order("created_at", desc=False)

        units_resp = execute_with_course_id_fallback("Rag_Unit", build_units_query, course_id)
        units = units_resp.data or []

        unit_ids: list[int] = []
        for u in units:
            uid = u.get("rag_unit_id")
            if uid is not None:
                try:
                    unit_ids.append(int(uid))
                except (TypeError, ValueError):
                    pass
        unit_ids = list(dict.fromkeys(unit_ids))
        quizzes_by_unit = _quizzes_by_rag_unit_ids(unit_ids, course_id=course_id)

        for unit in units:
            uid = unit.get("rag_unit_id")
            uid_int = int(uid) if uid is not None else None
            unit["quizzes"] = quizzes_by_unit.get(uid_int, []) if uid_int is not None else []

        units = to_json_safe(units)
        return {"units": units, "count": len(units)}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("GET /rag/units 錯誤")
        raise HTTPException(status_code=500, detail=f"列出 Rag_Unit 失敗: {e!s}")


def _fetch_rag_unit_row_by_id_or_http_error(
    *,
    owner_pid: str,
    rag_unit_id: int,
    rag_page_id: str,
    course_id,
    cols_with_folder: str,
    cols_without_folder: str,
    expected_unit_type: int,
    unit_type_desc: str,
) -> dict:
    """
    依 rag_unit_id＋擁有者查 Rag_Unit 一列並驗證（存在、未刪、rag_page_id 一致、unit_type 相符）。
    folder_combination 欄位不存在（42703）時改用 cols_without_folder 重查。
    """
    tab = (rag_page_id or "").strip()
    supabase = get_supabase()

    def fetch(cols_base: str):
        def build(with_course_filter: bool):
            cols = select_without_course_id_if_needed("Rag_Unit", cols_base, with_course_filter)
            q = (
                supabase.table("Rag_Unit")
                .select(cols)
                .eq("rag_unit_id", rag_unit_id)
                .eq("person_id", owner_pid)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        return execute_with_course_id_fallback("Rag_Unit", build, course_id)

    try:
        sel = fetch(cols_with_folder)
    except APIError as e:
        msg = (e.message or "").lower()
        if e.code == "42703" and "folder_combination" in msg:
            sel = fetch(cols_without_folder)
        else:
            raise
    if not sel.data:
        raise HTTPException(
            status_code=404,
            detail="找不到該 rag_unit_id，或與此 rag_page_id／擁有者不一致",
        )
    row = sel.data[0]
    if row.get("deleted"):
        raise HTTPException(status_code=404, detail="該單元已刪除")
    if (row.get("rag_page_id") or "").strip() != tab:
        raise HTTPException(
            status_code=400,
            detail="rag_page_id 與該 rag_unit_id 所屬之 Rag_Unit.rag_page_id 不一致",
        )
    try:
        ut = int(row.get("unit_type") or 0)
    except (TypeError, ValueError):
        ut = 0
    if ut != expected_unit_type:
        raise HTTPException(
            status_code=400,
            detail=f"僅 unit_type={unit_type_desc}可使用此端點，目前 unit_type={ut}",
        )
    return row


@router.get(
    "/units/{rag_unit_id}/mp3-file",
    summary="Rag Tab Unit Mp3 File",
    operation_id="rag_tab_unit_mp3_file",
    response_model=RagUnitMp3FileResponse,
)
def rag_tab_unit_mp3_file(
    course_id: CourseId,
    rag_unit_id: int = PathParam(..., gt=0, description="Rag_Unit 主鍵"),
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab；repack/upload 路徑皆在其下）"),
):
    """
    依 rag_page_id 與 rag_unit_id；**僅 Rag_Unit.unit_type=3（音訊單元）** 時回傳原始音訊。
    **不需** query `person_id`；後端依 `rag_page_id` 自 Rag 解析擁有者後讀 Storage。
    **優先**自該單元之 **repack** ZIP（`Rag_Unit.repack_file_name`／Storage `…/repack/{單元}.zip`）內，依 **folder_combination**（無則 **unit_name**）
    路徑段擷取第一個支援的音訊檔（repack 內仍保留上傳時之資料夾名，與 `repack_tasks_to_zips` 一致）。
    repack 無法讀取時**改讀**該 tab 之 **upload** ZIP（與 GET /rag/unit/mp3-file 相同）。
    Storage `…/rag/{tab}_rag.zip` 僅為逐字稿封包，不含原始 mp3。
    """
    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    row = _fetch_rag_unit_row_by_id_or_http_error(
        owner_pid=owner_pid,
        rag_unit_id=rag_unit_id,
        rag_page_id=rag_page_id,
        course_id=course_id,
        cols_with_folder="rag_unit_id, rag_page_id, unit_name, folder_combination, unit_type, deleted, repack_file_name, transcript, course_id",
        cols_without_folder="rag_unit_id, rag_page_id, unit_name, unit_type, deleted, repack_file_name, transcript, course_id",
        expected_unit_type=RAG_UNIT_TYPE_MP3,
        unit_type_desc="3（mp3 音訊單元）",
    )
    folder_name = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if not folder_name:
        raise HTTPException(
            status_code=400,
            detail="Rag_Unit.folder_combination 與 unit_name 皆為空，無法對應 repack／upload ZIP 內單元路徑",
        )

    zip_bytes: bytes | None = None
    zip_is_unit_repack = False
    repack_fn = (row.get("repack_file_name") or "").strip()
    repack_err: str | None = None
    if repack_fn:
        try:
            zip_bytes = read_repack_zip_bytes(repack_fn)
            zip_is_unit_repack = True
        except FileNotFoundError as e:
            repack_err = str(e)
        except ValueError as e:
            repack_err = str(e)
        except Exception as e:
            _logger.exception("讀取 repack ZIP 失敗")
            repack_err = str(e)

    if zip_bytes is None:
        try:
            zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            zip_is_unit_repack = False
        except FileNotFoundError as e:
            detail = str(e)
            if repack_err:
                detail = f"{detail}（repack 亦失敗：{repack_err}）"
            raise HTTPException(status_code=404, detail=detail) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _logger.exception("讀取 upload ZIP 失敗")
            raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e

    from_repack_ok = bool(repack_fn) and repack_err is None
    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip_with_folder_fallback(
            zip_bytes,
            folder_name,
            allow_scan_other_top_folders=zip_is_unit_repack,
        )
    except ValueError as e:
        if from_repack_ok and zip_is_unit_repack:
            try:
                zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            except FileNotFoundError as e2:
                raise HTTPException(
                    status_code=400,
                    detail=f"{e!s}（repack ZIP 內亦無法對應音訊，且無 upload 可備援：{e2!s}）",
                ) from e
            except ValueError as e2:
                raise HTTPException(status_code=400, detail=f"{e!s}（upload 備援：{e2!s}）") from e
            except Exception as e2:
                _logger.exception("讀取 upload ZIP 備援失敗")
                raise HTTPException(
                    status_code=500,
                    detail=f"{e!s}（upload 備援讀取失敗：{e2!s}）",
                ) from e
            try:
                contents, suffix, inner_path = pick_audio_from_upload_zip_with_folder_fallback(
                    zip_bytes,
                    folder_name,
                    allow_scan_other_top_folders=False,
                )
            except ValueError as e3:
                raise HTTPException(status_code=400, detail=str(e3)) from e
        else:
            raise HTTPException(status_code=400, detail=str(e)) from e

    media = audio_media_type_for_suffix(suffix)
    disp_name = Path(inner_path).name
    audio_b64 = base64.b64encode(contents).decode()
    unit_transcript = transcript_from_row(row)
    return RagUnitMp3FileResponse(
        rag_unit_id=rag_unit_id,
        rag_page_id=tab,
        audio_base64=audio_b64,
        media_type=media,
        filename=disp_name,
        transcript=unit_transcript,
    )


@router.get(
    "/units/{rag_unit_id}/text",
    summary="Rag Tab Unit Text",
    operation_id="rag_tab_unit_text",
    response_model=RagUnitTextResponse,
)
def rag_tab_unit_text(
    course_id: CourseId,
    rag_unit_id: int = PathParam(..., gt=0, description="Rag_Unit 主鍵"),
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab）"),
):
    """
    依 rag_page_id 與 rag_unit_id；**僅 Rag_Unit.unit_type=2（文字單元）** 時回傳 `text_file_name` 與 `transcript`（全文，含 Markdown）。
    **不需** query `person_id`；後端依 `rag_page_id` 自 Rag 解析擁有者。
    逐字稿以 `Rag_Unit.transcript` 為準；DB 無逐字稿時改讀該 tab 之 **upload** ZIP（以 `folder_combination` 或 `unit_name` 為資料夾名，與 GET /rag/unit/text 相同）。
    取代 deprecated 之 GET /rag/unit/text。
    """
    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    row = _fetch_rag_unit_row_by_id_or_http_error(
        owner_pid=owner_pid,
        rag_unit_id=rag_unit_id,
        rag_page_id=rag_page_id,
        course_id=course_id,
        cols_with_folder="rag_unit_id, rag_page_id, unit_name, folder_combination, unit_type, deleted, text_file_name, transcript, course_id",
        cols_without_folder="rag_unit_id, rag_page_id, unit_name, unit_type, deleted, text_file_name, transcript, course_id",
        expected_unit_type=RAG_UNIT_TYPE_TEXT,
        unit_type_desc="2（文字單元）",
    )

    text_file_name = (row.get("text_file_name") or "").strip()
    transcript = transcript_from_row(row)
    zip_folder = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if not transcript and zip_folder:
        try:
            zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            transcript, inner_path = read_single_transcript_text_from_upload_zip(zip_bytes, zip_folder)
            if not text_file_name:
                text_file_name = Path(inner_path).name
        except (FileNotFoundError, ValueError) as e:
            _logger.debug("GET /rag/units/{id}/text ZIP 備援略過: %s", e)
        except Exception:
            _logger.exception("GET /rag/units/{id}/text ZIP 備援失敗")

    if not transcript.strip():
        raise HTTPException(
            status_code=404,
            detail="該單元無逐字稿（Rag_Unit.transcript 為空，且自 upload ZIP 讀取備援失敗或內容為空）",
        )

    return RagUnitTextResponse(
        rag_unit_id=rag_unit_id,
        rag_page_id=tab,
        folder_name=zip_folder,
        text_file_name=text_file_name,
        transcript=transcript,
    )


@router.get(
    "/units/{rag_unit_id}/youtube-url",
    summary="Rag Tab Unit Youtube Url",
    operation_id="rag_tab_unit_youtube_url",
    response_model=RagUnitYoutubeUrlResponse,
)
def rag_tab_unit_youtube_url(
    course_id: CourseId,
    rag_unit_id: int = PathParam(..., gt=0, description="Rag_Unit 主鍵"),
    rag_page_id: str = Query(..., description="Rag.rag_page_id（parent tab）"),
):
    """
    依 rag_page_id 與 rag_unit_id；**僅 Rag_Unit.unit_type=4（YouTube 單元）** 時回傳 watch URL 與 `transcript`。
    **不需** query `person_id`；後端依 `rag_page_id` 自 Rag 解析擁有者。
    `youtube_url`／`transcript` 以 `Rag_Unit` 欄位為準；DB 缺值時改讀該 tab 之 **upload** ZIP
    （文字檔第一行為 YouTube URL、第二行起為逐字稿，與 GET /rag/unit/youtube-url 相同）。
    取代 deprecated 之 GET /rag/unit/youtube-url。
    """
    owner_pid = resolve_rag_tab_owner_person_id(rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    row = _fetch_rag_unit_row_by_id_or_http_error(
        owner_pid=owner_pid,
        rag_unit_id=rag_unit_id,
        rag_page_id=rag_page_id,
        course_id=course_id,
        cols_with_folder="rag_unit_id, rag_page_id, unit_name, folder_combination, unit_type, deleted, text_file_name, youtube_url, transcript, course_id",
        cols_without_folder="rag_unit_id, rag_page_id, unit_name, unit_type, deleted, text_file_name, youtube_url, transcript, course_id",
        expected_unit_type=RAG_UNIT_TYPE_YOUTUBE,
        unit_type_desc="4（YouTube 單元）",
    )

    youtube_url = (row.get("youtube_url") or "").strip()
    text_file_name = (row.get("text_file_name") or "").strip()
    transcript = transcript_from_row(row)
    zip_folder = (row.get("folder_combination") or row.get("unit_name") or "").strip()
    if (not youtube_url or not transcript) and zip_folder:
        try:
            zip_bytes = read_upload_zip_bytes(owner_pid, rag_page_id)
            if not youtube_url:
                vid, inner_path = read_youtube_video_id_from_upload_zip(zip_bytes, zip_folder)
                youtube_url = f"https://www.youtube.com/watch?v={vid}"
                if not text_file_name:
                    text_file_name = Path(inner_path).name
            if not transcript:
                transcript, inner_path = read_supplementary_text_from_youtube_unit(zip_bytes, zip_folder)
                if not text_file_name:
                    text_file_name = Path(inner_path).name
        except (FileNotFoundError, ValueError) as e:
            _logger.debug("GET /rag/units/{id}/youtube-url ZIP 備援略過: %s", e)
        except Exception:
            _logger.exception("GET /rag/units/{id}/youtube-url ZIP 備援失敗")

    if not youtube_url:
        raise HTTPException(
            status_code=404,
            detail="該單元無 youtube_url，且無法自 upload ZIP 解析（文字檔第一行須為 YouTube URL）",
        )
    if not transcript.strip():
        raise HTTPException(
            status_code=404,
            detail="該單元無逐字稿（Rag_Unit.transcript 為空，且自 upload ZIP 讀取備援失敗或第二行起無內容）",
        )

    return RagUnitYoutubeUrlResponse(
        rag_unit_id=rag_unit_id,
        rag_page_id=tab,
        folder_name=zip_folder,
        youtube_url=youtube_url,
        text_file_name=text_file_name,
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# GET /rag/pages/{rag_page_id}/unit-preview/*（建置前，Rag_Unit 尚未存在時用）
# ---------------------------------------------------------------------------


def _read_upload_zip_bytes_or_http_error(person_id: str, rag_page_id: str) -> bytes:
    """讀取 upload ZIP 內容；對應 404（找不到）／400（值錯誤）／500（其他）HTTPException。"""
    try:
        return read_upload_zip_bytes(person_id, rag_page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _logger.exception("讀取 upload ZIP 失敗")
        raise HTTPException(status_code=500, detail=f"讀取 upload ZIP 失敗: {e!s}") from e


@router.get(
    "/pages/{rag_page_id}/unit-preview/text",
    summary="Rag Page Unit Preview Text",
    operation_id="rag_page_unit_preview_text",
    response_model=RagUnitTextPreviewResponse,
)
def rag_page_unit_preview_text(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(..., description="upload ZIP 內單元資料夾名"),
):
    """
    **建置前預覽**（Rag_Unit 尚未建立、無 rag_unit_id 時用）：自 upload ZIP 內指定資料夾讀取**恰好一個**文字檔全文作為 `transcript`（unit_type=2，與 build-rag-zip 一致）。
    呼叫者（Bearer token）須為該 `rag_page_id` 之 Rag.person_id。
    已建置之單元請改用 GET /rag/units/{rag_unit_id}/text。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        transcript, inner_path = read_single_transcript_text_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail=f"「{Path(inner_path).name}」內容為空（unit_type=2 文字單元逐字稿不可為空）",
        )

    return RagUnitTextPreviewResponse(
        rag_page_id=tab,
        folder_name=folder,
        text_file_name=Path(inner_path).name,
        transcript=transcript,
    )


@router.get(
    "/pages/{rag_page_id}/unit-preview/mp3-file",
    summary="Rag Page Unit Preview Mp3 File",
    operation_id="rag_page_unit_preview_mp3_file",
    response_model=RagUnitMp3FilePreviewResponse,
)
def rag_page_unit_preview_mp3_file(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(..., description="upload ZIP 內單元資料夾名"),
):
    """
    **建置前預覽**：自 upload ZIP 內指定資料夾擷取音訊（base64）與**恰好一個**文字檔全文作為 `transcript`（unit_type=3，須音訊＋逐字稿，與 build-rag-zip 一致）。
    呼叫者（Bearer token）須為該 `rag_page_id` 之 Rag.person_id。**永遠回 JSON**。
    已建置之單元請改用 GET /rag/units/{rag_unit_id}/mp3-file。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        contents, suffix, inner_path = pick_audio_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        transcript, text_file_name = read_mp3_unit_transcript_from_upload_zip(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RagUnitMp3FilePreviewResponse(
        rag_page_id=tab,
        folder_name=folder,
        audio_base64=base64.b64encode(contents).decode(),
        media_type=audio_media_type_for_suffix(suffix),
        filename=Path(inner_path).name,
        text_file_name=text_file_name,
        transcript=transcript,
    )


@router.get(
    "/pages/{rag_page_id}/unit-preview/youtube-url",
    summary="Rag Page Unit Preview Youtube Url",
    operation_id="rag_page_unit_preview_youtube_url",
    response_model=RagUnitYoutubeUrlPreviewResponse,
)
def rag_page_unit_preview_youtube_url(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_page_id: str = PathParam(..., description="Rag.rag_page_id（upload ZIP 路徑）"),
    folder_name: str = Query(..., description="upload ZIP 內單元資料夾名"),
):
    """
    **建置前預覽**：自 upload ZIP 內指定資料夾讀取**恰好一個**文字檔：第一行為 YouTube URL，第二行起為 `transcript`（unit_type=4，與 build-rag-zip 一致）。
    呼叫者（Bearer token）須為該 `rag_page_id` 之 Rag.person_id。
    已建置之單元請改用 GET /rag/units/{rag_unit_id}/youtube-url。
    """
    require_rag_tab_owner(caller_person_id, rag_page_id, course_id)
    tab = (rag_page_id or "").strip()
    folder = (folder_name or "").strip()
    zip_bytes = _read_upload_zip_bytes_or_http_error(caller_person_id, rag_page_id)

    try:
        vid, inner_path = read_youtube_video_id_from_upload_zip(zip_bytes, folder)
        transcript, _ = read_supplementary_text_from_youtube_unit(zip_bytes, folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail=f"「{Path(inner_path).name}」第二行起無逐字稿內容（unit_type=4 須第一行 YouTube URL、第二行起逐字稿）",
        )

    return RagUnitYoutubeUrlPreviewResponse(
        rag_page_id=tab,
        folder_name=folder,
        youtube_url=f"https://www.youtube.com/watch?v={vid}",
        text_file_name=Path(inner_path).name,
        transcript=transcript,
    )


@router.post("/quizzes", status_code=201, summary="Rag Create Quiz (no LLM)", operation_id="rag_create_quiz")
def insert_rag_quiz_row(
    body: openapi_body(InsertRagQuizRowRequest, {"rag_page_id": "string", "rag_unit_id": 1}),
    caller_person_id: PersonId,
    course_id: CourseId,
):
    """
    依 `rag_page_id`／`rag_unit_id` 解析 Rag_Unit 後新增一筆空白 `Rag_Quiz`，**不呼叫 LLM**。`rag_quiz_id` 由資料庫自動產生並於回傳中帶出。
    LLM 出題請用 answer router 之 LLM 出題端點。
    """
    try:
        supabase = get_supabase()
        req_tab = (body.rag_page_id or "").strip()
        resolved_unit_id = int(body.rag_unit_id or 0)

        u: dict[str, Any] | None = None

        def build_unit_lookup(with_course_filter: bool, *, by_unit_id: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Unit",
                "rag_unit_id, rag_page_id, person_id, unit_name, course_id",
                with_course_filter,
            )
            q = supabase.table("Rag_Unit").select(cols).eq("deleted", False)
            if by_unit_id:
                q = q.eq("rag_unit_id", resolved_unit_id).limit(1)
            else:
                q = q.eq("rag_page_id", req_tab)
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q

        if resolved_unit_id > 0:
            sel = execute_with_course_id_fallback(
                "Rag_Unit",
                lambda with_course: build_unit_lookup(with_course, by_unit_id=True),
                course_id,
            )
            if sel.data:
                u = sel.data[0]
        else:
            if not req_tab:
                raise HTTPException(
                    status_code=400,
                    detail="請傳入 rag_unit_id（>0），或傳入 rag_page_id 且該 tab 下僅有一筆 Rag_Unit",
                )
            sel = execute_with_course_id_fallback(
                "Rag_Unit",
                lambda with_course: build_unit_lookup(with_course, by_unit_id=False),
                course_id,
            )
            rows = sel.data or []
            if len(rows) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"rag_page_id 需唯一對應一筆 Rag_Unit（目前 {len(rows)} 筆），請改傳 rag_unit_id",
                )
            u = rows[0]

        if u is None:
            raise HTTPException(status_code=404, detail="找不到該 rag_unit_id 的 Rag_Unit 資料，或已刪除")

        uid = int(u.get("rag_unit_id") or 0)
        if req_tab and (u.get("rag_page_id") or "").strip() != req_tab:
            raise HTTPException(status_code=400, detail="rag_page_id 與 rag_unit_id 對應之 Rag_Unit 不一致")
        pid = (u.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權於該 Rag_Unit 新增題目")
        rag_page_id = (u.get("rag_page_id") or "").strip()
        if not rag_page_id:
            raise HTTPException(status_code=400, detail="該 Rag_Unit 的 rag_page_id 為空，無法寫入 Rag_Quiz")
        quiz_name = (u.get("unit_name") or "").strip()
        ts = now_taipei_iso()
        quiz_row: dict[str, Any] = {
            "rag_page_id": rag_page_id,
            "rag_unit_id": uid,
            "person_id": pid,
            "course_id": course_id,
            "quiz_name": quiz_name,
            "quiz_user_prompt_text": "",
            "quiz_content": "",
            "quiz_hint": "",
            "quiz_answer_reference": "",
            "answer_user_prompt_text": "",
            "answer_content": "",
            "answer_critique": None,
            "for_exam": False,
            "follow_up": False,
            "deleted": False,
            "updated_at": ts,
            "created_at": ts,
        }
        ins = insert_rag_child_row("Rag_Quiz", quiz_row)
        if not ins.data or len(ins.data) == 0:
            raise HTTPException(status_code=500, detail="寫入 Rag_Quiz 失敗（無回傳資料）")
        row = ins.data[0]
        ans = (row.get("quiz_answer") or row.get("answer_content") or "") or ""
        return to_json_safe(
            {
                "rag_quiz_id": row.get("rag_quiz_id"),
                "rag_page_id": row.get("rag_page_id"),
                "rag_unit_id": row.get("rag_unit_id"),
                "person_id": row.get("person_id"),
                "quiz_name": row.get("quiz_name"),
                "quiz_user_prompt_text": row.get("quiz_user_prompt_text"),
                "quiz_content": row.get("quiz_content"),
                "quiz_hint": row.get("quiz_hint"),
                "quiz_answer_reference": row.get("quiz_answer_reference"),
                "answer_user_prompt_text": row.get("answer_user_prompt_text"),
                "quiz_answer": ans,
                "answer_content": ans,
                "answer_critique": row.get("answer_critique"),
                "for_exam": row.get("for_exam"),
                "follow_up": row.get("follow_up"),
                "deleted": row.get("deleted"),
                "updated_at": row.get("updated_at"),
                "created_at": row.get("created_at"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("POST /rag/quizzes 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/quizzes/{rag_quiz_id}", summary="Update Rag Quiz Name", operation_id="rag_tab_unit_quiz_quiz_name")
def update_rag_quiz_name(
    body: openapi_body(UpdateRagQuizQuizNameRequest, {"quiz_name": "新名稱"}),
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_quiz_id: int = PathParam(..., gt=0, description="要更新的 Rag_Quiz 主鍵"),
):
    """
    更新既有 Rag_Quiz 的 quiz_name。以 rag_quiz_id（主鍵）比對；僅更新 deleted=false 的列。
    回傳 rag_quiz_id、rag_page_id、rag_unit_id、person_id、quiz_name、updated_at。
    """
    quiz_name = (body.quiz_name or "").strip()
    if not quiz_name:
        raise HTTPException(status_code=400, detail="請傳入 quiz_name")
    try:
        supabase = get_supabase()

        def build_quiz_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz 資料，或已刪除")
        row = sel.data[0]
        pid = (row.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權修改該 Rag_Quiz")
        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"quiz_name": quiz_name, "updated_at": ts}).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        return {
            "rag_quiz_id": rag_quiz_id,
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": pid,
            "quiz_name": quiz_name,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("PATCH /rag/quizzes/{rag_quiz_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/quizzes/{rag_quiz_id}",
    status_code=200,
    summary="Delete Rag Quiz",
    operation_id="rag_tab_unit_quiz_delete",
)
def delete_rag_quiz(
    caller_person_id: PersonId,
    course_id: CourseId,
    rag_quiz_id: int = PathParam(..., gt=0, description="要軟刪除的 Rag_Quiz 主鍵"),
):
    """
    DELETE /rag/quizzes/{rag_quiz_id}。
    軟刪除：將 Rag_Quiz 該列 deleted 設為 true（僅 person_id 與請求者一致且尚未刪除之列）。
    """
    try:
        supabase = get_supabase()

        def build_quiz_delete_sel(with_course_filter: bool):
            cols = select_without_course_id_if_needed(
                "Rag_Quiz",
                "rag_quiz_id, rag_page_id, rag_unit_id, person_id, course_id",
                with_course_filter,
            )
            q = (
                supabase.table("Rag_Quiz")
                .select(cols)
                .eq("rag_quiz_id", rag_quiz_id)
                .eq("deleted", False)
            )
            if with_course_filter and course_id is not None:
                q = q.eq("course_id", course_id)
            return q.limit(1)

        sel = execute_with_course_id_fallback("Rag_Quiz", build_quiz_delete_sel, course_id)
        if not sel.data:
            raise HTTPException(status_code=404, detail="找不到該 rag_quiz_id 的 Rag_Quiz 資料，或已刪除")
        row = sel.data[0]
        pid = (row.get("person_id") or "").strip()
        if pid != caller_person_id:
            raise HTTPException(status_code=403, detail="無權刪除該 Rag_Quiz")
        ts = now_taipei_iso()
        supabase.table("Rag_Quiz").update({"deleted": True, "updated_at": ts}).eq("rag_quiz_id", rag_quiz_id).eq("deleted", False).execute()
        return {
            "message": "已將 Rag_Quiz 標記為刪除",
            "rag_quiz_id": rag_quiz_id,
            "rag_page_id": row.get("rag_page_id"),
            "rag_unit_id": row.get("rag_unit_id"),
            "person_id": pid,
            "rag_quiz_updated": True,
            "updated_at": ts,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("DELETE /rag/quizzes/{rag_quiz_id} 錯誤")
        raise HTTPException(status_code=500, detail=str(e))
