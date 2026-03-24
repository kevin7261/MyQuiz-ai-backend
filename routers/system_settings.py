"""
系統設定相關 API 模組。
- GET /system-settings/course-name：取得 course_name 設定（key=course_name），直接取值，無參數。
- GET /system-settings/llm-api-key：取得預設 LLM API Key（key=llm_api_key），直接取值，無參數。
- PUT /system-settings/course-name：寫入 course_name（key=course_name，value=body.course_name）。
- PUT /system-settings/llm-api-key：寫入預設 LLM API Key（key=llm_api_key）。
- GET/PUT /system-settings/rag-for-exam-localhost：key=rag_localhost，value=rag_id（已有則覆蓋）。
- GET/PUT /system-settings/rag-for-exam-deploy：key=rag_deploy，value=rag_id（已有則覆蓋）。
"""

# 引入 Optional 型別
from typing import Optional

# 引入 FastAPI 與 HTTPException
from fastapi import APIRouter, HTTPException
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 引入 UTC 時間工具
from utils.datetime_utils import now_utc_iso
# 引入 Supabase 客戶端
from utils.supabase_client import get_supabase
# 供測驗 RAG：System_Setting 固定 key
from utils.rag_exam_setting import RAG_EXAM_SETTING_KEY_DEPLOY, RAG_EXAM_SETTING_KEY_LOCALHOST

# 建立路由，前綴為 /system-settings，標籤為 system-settings
router = APIRouter(prefix="/system-settings", tags=["system-settings"])

# System_Setting 表：key / value。llm_api_key、course_name 各一筆，直接取值。
SYSTEM_SETTING_LLM_KEY = "llm_api_key"
SYSTEM_SETTING_COURSE_NAME_KEY = "course_name"
SYSTEM_SETTING_COLUMNS = "system_setting_id, key, value, updated_at, created_at"


class CourseNameResponse(BaseModel):
    """GET/PUT /system-settings/course-name 回應。key=course_name 的 value 回傳在 course_name 欄位。"""
    system_setting_id: Optional[int] = None
    course_name: Optional[str] = None  # 對應 System_Setting.value
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class LlmApiKeyResponse(BaseModel):
    """GET/PUT /system-settings/llm-api-key 回應。key=llm_api_key 的 value 回傳在 llm_api_key 欄位。"""
    llm_api_key_id: Optional[int] = None  # 對應 system_setting_id
    llm_api_key: Optional[str] = None     # 對應 value
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class PutCourseNameRequest(BaseModel):
    """PUT /system-settings/course-name 的 body：僅傳 course_name。"""
    course_name: str = Field(..., description="課程名稱，對應 System_Setting.key")


class PutLlmApiKeyRequest(BaseModel):
    """PUT /system-settings/llm-api-key 的 body：僅傳 llm_api_key。"""
    llm_api_key: str = Field(..., description="LLM API Key，可為空字串表示清除")


class PutRagForExamRequest(BaseModel):
    """PUT rag-for-exam-localhost / rag-for-exam-deploy：Rag 表主鍵 rag_id，寫入 System_Setting.value（字串）。"""
    rag_id: int = Field(..., description="Rag 表 rag_id，存為 System_Setting.value（字串）")


class RagForExamSettingResponse(BaseModel):
    """GET/PUT rag-for-exam-localhost、rag-for-exam-deploy 回應。"""
    key: str
    rag_id: Optional[int] = None
    system_setting_id: Optional[int] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


@router.get("/course-name", response_model=CourseNameResponse)
def get_course_name_setting():
    """
    取得 course_name 設定（System_Setting 表 key=course_name）。直接取值，無參數。
    回傳 course_name 欄位（對應該 key 的 value）。若尚無資料，皆為 null。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_COURSE_NAME_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return CourseNameResponse()
        row = resp.data[0]
        return CourseNameResponse(
            system_setting_id=row.get("system_setting_id"),
            course_name=row.get("value"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-api-key", response_model=LlmApiKeyResponse)
def get_llm_api_key():
    """
    取得系統預設的 LLM API Key（System_Setting 表 key=llm_api_key）。
    若尚無資料，回傳 llm_api_key_id 等皆為 null。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_LLM_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return LlmApiKeyResponse()
        row = resp.data[0]
        return LlmApiKeyResponse(
            llm_api_key_id=row.get("system_setting_id"),
            llm_api_key=row.get("value"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _upsert_setting_and_get_row(supabase, key: str, value: str):
    """依 key 新增或更新一筆 System_Setting，回傳該筆 row（dict）。"""
    now = now_utc_iso()
    resp = (
        supabase.table("System_Setting")
        .select("system_setting_id")
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if resp.data and len(resp.data) > 0:
        row_id = resp.data[0].get("system_setting_id")
        supabase.table("System_Setting").update({
            "value": value,
            "updated_at": now,
        }).eq("system_setting_id", row_id).execute()
    else:
        supabase.table("System_Setting").insert({
            "key": key,
            "value": value,
            "updated_at": now,
            "created_at": now,
        }).execute()
    resp2 = (
        supabase.table("System_Setting")
        .select(SYSTEM_SETTING_COLUMNS)
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if not resp2.data or len(resp2.data) == 0:
        return None
    return resp2.data[0]


@router.put("/course-name", response_model=CourseNameResponse)
def put_course_name_setting(body: PutCourseNameRequest):
    """
    寫入 course_name 設定（System_Setting 表 key=course_name，value=body.course_name）。
    已有則更新，否則新增。回傳 course_name 欄位。
    """
    value_to_save = (body.course_name or "").strip() or ""
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, SYSTEM_SETTING_COURSE_NAME_KEY, value_to_save)
        if not row:
            return CourseNameResponse(course_name=value_to_save or None)
        return CourseNameResponse(
            system_setting_id=row.get("system_setting_id"),
            course_name=row.get("value"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-api-key", response_model=LlmApiKeyResponse)
def put_llm_api_key(body: PutLlmApiKeyRequest):
    """
    寫入或更新系統預設的 LLM API Key（System_Setting 表 key=llm_api_key）。
    已有則更新，否則新增。llm_api_key 可傳空字串表示清除。
    """
    value_to_save = (body.llm_api_key or "").strip() or ""
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, SYSTEM_SETTING_LLM_KEY, value_to_save)
        if not row:
            return LlmApiKeyResponse(llm_api_key=value_to_save or None)
        return LlmApiKeyResponse(
            llm_api_key_id=row.get("system_setting_id"),
            llm_api_key=row.get("value"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_rag_for_exam_setting_row(supabase, key: str) -> RagForExamSettingResponse:
    resp = (
        supabase.table("System_Setting")
        .select(SYSTEM_SETTING_COLUMNS)
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if not resp.data or len(resp.data) == 0:
        return RagForExamSettingResponse(key=key, rag_id=None)
    row = resp.data[0]
    raw = (row.get("value") or "").strip()
    rid: Optional[int] = None
    if raw:
        try:
            rid = int(raw)
        except ValueError:
            rid = None
    return RagForExamSettingResponse(
        key=key,
        rag_id=rid,
        system_setting_id=row.get("system_setting_id"),
        updated_at=row.get("updated_at"),
        created_at=row.get("created_at"),
    )


def _put_rag_for_exam_for_key(key: str, body: PutRagForExamRequest) -> RagForExamSettingResponse:
    if body.rag_id <= 0:
        raise HTTPException(status_code=400, detail="無效的 rag_id")
    value_to_save = str(body.rag_id)
    try:
        supabase = get_supabase()
        chk = (
            supabase.table("Rag")
            .select("rag_id")
            .eq("rag_id", body.rag_id)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not chk.data or len(chk.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該 rag_id 的 Rag 資料")
        row = _upsert_setting_and_get_row(supabase, key, value_to_save)
        if not row:
            return RagForExamSettingResponse(key=key, rag_id=body.rag_id)
        return RagForExamSettingResponse(
            key=key,
            rag_id=body.rag_id,
            system_setting_id=row.get("system_setting_id"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-for-exam-localhost", response_model=RagForExamSettingResponse)
def get_rag_for_exam_localhost():
    """讀取 key=rag_localhost 的供測驗 RAG rag_id（value 轉 int）。無資料則 rag_id 為 null。"""
    try:
        supabase = get_supabase()
        return _get_rag_for_exam_setting_row(supabase, RAG_EXAM_SETTING_KEY_LOCALHOST)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-for-exam-deploy", response_model=RagForExamSettingResponse)
def get_rag_for_exam_deploy():
    """讀取 key=rag_deploy 的供測驗 RAG rag_id（value 轉 int）。無資料則 rag_id 為 null。"""
    try:
        supabase = get_supabase()
        return _get_rag_for_exam_setting_row(supabase, RAG_EXAM_SETTING_KEY_DEPLOY)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rag-for-exam-localhost", response_model=RagForExamSettingResponse)
def put_rag_for_exam_localhost(body: PutRagForExamRequest):
    """
    寫入供測驗 RAG（System_Setting key=rag_localhost，value=str(rag_id)）。
    該 key 已存在則更新，否則新增。需存在 rag_id 且 Rag.deleted=false。
    """
    return _put_rag_for_exam_for_key(RAG_EXAM_SETTING_KEY_LOCALHOST, body)


@router.put("/rag-for-exam-deploy", response_model=RagForExamSettingResponse)
def put_rag_for_exam_deploy(body: PutRagForExamRequest):
    """
    寫入供測驗 RAG（System_Setting key=rag_deploy，value=str(rag_id)）。
    該 key 已存在則更新，否則新增。需存在 rag_id 且 Rag.deleted=false。
    """
    return _put_rag_for_exam_for_key(RAG_EXAM_SETTING_KEY_DEPLOY, body)
