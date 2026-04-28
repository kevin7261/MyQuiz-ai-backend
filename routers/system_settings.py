"""
系統設定相關 API 模組。
- GET /system-settings/course-name：取得 course_name 設定（key=course_name），直接取值，無參數。
- GET /system-settings/llm-api-key：取得預設 LLM API Key（key=llm_api_key），直接取值，無參數。
- GET /system-settings/deepgram-api-key：取得 Deepgram API Key（key=deepgram_api_key），直接取值，無參數。
- PUT /system-settings/course-name：寫入 course_name（key=course_name，value=body.course_name）。
- PUT /system-settings/llm-api-key：寫入預設 LLM API Key（key=llm_api_key）。
- PUT /system-settings/deepgram-api-key：寫入 Deepgram API Key（key=deepgram_api_key）。
- GET/PUT /system-settings/english-system-for-exam-localhost：key=english_system_localhost，value=English_System.system_id；PUT 可傳空字串清除。
- GET/PUT /system-settings/english-system-for-exam-deploy：key=english_system_deploy，同上。
"""

# 引入 Optional、Union 型別
from typing import Optional, Union

# 引入 FastAPI 與 HTTPException
from fastapi import APIRouter, HTTPException

from dependencies.person_id import PersonId
# 引入 Pydantic 的 BaseModel、Field
from pydantic import BaseModel, Field

# 引入台北時間工具
from utils.datetime_utils import now_taipei_iso, to_taipei_iso
# 引入 Supabase 客戶端
from utils.supabase_client import get_supabase
# 供測驗 RAG：System_Setting 固定 key
from utils.english_system_exam_setting import (
    ENGLISH_SYSTEM_EXAM_SETTING_KEY_DEPLOY,
    ENGLISH_SYSTEM_EXAM_SETTING_KEY_LOCALHOST,
)

# 建立路由，前綴為 /system-settings，標籤為 system-settings
router = APIRouter(prefix="/system-settings", tags=["system-settings"])

# System_Setting 表：key / value。llm_api_key、deepgram_api_key、course_name 各一筆，直接取值。
SYSTEM_SETTING_LLM_KEY = "llm_api_key"
SYSTEM_SETTING_DEEPGRAM_KEY = "deepgram_api_key"
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


class DeepgramApiKeyResponse(BaseModel):
    """GET/PUT /system-settings/deepgram-api-key 回應。key=deepgram_api_key 的 value 回傳在 deepgram_api_key 欄位。"""
    deepgram_api_key_id: Optional[int] = None
    deepgram_api_key: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class PutCourseNameRequest(BaseModel):
    """PUT /system-settings/course-name 的 body：僅傳 course_name。"""
    course_name: str = Field(..., description="課程名稱，對應 System_Setting.key")


class PutLlmApiKeyRequest(BaseModel):
    """PUT /system-settings/llm-api-key 的 body：僅傳 llm_api_key。"""
    llm_api_key: str = Field(..., description="LLM API Key，可為空字串表示清除")


class PutDeepgramApiKeyRequest(BaseModel):
    """PUT /system-settings/deepgram-api-key 的 body：僅傳 deepgram_api_key。"""
    deepgram_api_key: str = Field(..., description="Deepgram API Key，可為空字串表示清除")


class PutEnglishSystemForExamRequest(BaseModel):
    """PUT english-system-for-exam-localhost / english-system-for-exam-deploy：正整數寫入 value；空字串清除該 key 的 value。"""
    system_id: Union[int, str] = Field(
        ...,
        description="English_System 表 system_id（主鍵，正整數），存為 System_Setting.value；傳空字串表示清除",
    )


class EnglishSystemForExamSettingResponse(BaseModel):
    """GET/PUT english-system-for-exam-localhost、english-system-for-exam-deploy 回應。"""
    key: str
    system_id: Optional[int] = None
    system_setting_id: Optional[int] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


@router.get("/course-name", response_model=CourseNameResponse)
def get_course_name_setting(_person_id: PersonId):
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
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-api-key", response_model=LlmApiKeyResponse)
def get_llm_api_key(_person_id: PersonId):
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
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deepgram-api-key", response_model=DeepgramApiKeyResponse)
def get_deepgram_api_key_setting(_person_id: PersonId):
    """
    取得 Deepgram API Key（System_Setting 表 key=deepgram_api_key）。
    若尚無資料，回傳 deepgram_api_key_id 等皆為 null。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_DEEPGRAM_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return DeepgramApiKeyResponse()
        row = resp.data[0]
        return DeepgramApiKeyResponse(
            deepgram_api_key_id=row.get("system_setting_id"),
            deepgram_api_key=row.get("value"),
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _upsert_setting_and_get_row(supabase, key: str, value: str):
    """依 key 新增或更新一筆 System_Setting，回傳該筆 row（dict）。"""
    now = now_taipei_iso()
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
def put_course_name_setting(body: PutCourseNameRequest, _person_id: PersonId):
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
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-api-key", response_model=LlmApiKeyResponse)
def put_llm_api_key(body: PutLlmApiKeyRequest, _person_id: PersonId):
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
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/deepgram-api-key", response_model=DeepgramApiKeyResponse)
def put_deepgram_api_key_setting(body: PutDeepgramApiKeyRequest, _person_id: PersonId):
    """
    寫入或更新 Deepgram API Key（System_Setting 表 key=deepgram_api_key）。
    已有則更新，否則新增。deepgram_api_key 可傳空字串表示清除。
    """
    value_to_save = (body.deepgram_api_key or "").strip() or ""
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, SYSTEM_SETTING_DEEPGRAM_KEY, value_to_save)
        if not row:
            return DeepgramApiKeyResponse(deepgram_api_key=value_to_save or None)
        return DeepgramApiKeyResponse(
            deepgram_api_key_id=row.get("system_setting_id"),
            deepgram_api_key=row.get("value"),
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_english_system_for_exam_setting_row(
    supabase, key: str
) -> EnglishSystemForExamSettingResponse:
    resp = (
        supabase.table("System_Setting")
        .select(SYSTEM_SETTING_COLUMNS)
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if not resp.data or len(resp.data) == 0:
        return EnglishSystemForExamSettingResponse(key=key, system_id=None)
    row = resp.data[0]
    raw = (row.get("value") or "").strip()
    sid: Optional[int] = None
    if raw:
        try:
            sid = int(raw)
        except ValueError:
            sid = None
    return EnglishSystemForExamSettingResponse(
        key=key,
        system_id=sid,
        system_setting_id=row.get("system_setting_id"),
        updated_at=to_taipei_iso(row.get("updated_at")),
        created_at=to_taipei_iso(row.get("created_at")),
    )


def _parse_english_system_id_for_put(raw: Union[int, str]) -> Optional[int]:
    """回傳正整數 English_System.system_id；空字串表示清除。其餘無效則拋 HTTPException。"""
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return None
        try:
            n = int(s)
        except ValueError:
            raise HTTPException(status_code=400, detail="system_id 必須為整數或空字串")
        if n <= 0:
            raise HTTPException(status_code=400, detail="無效的 system_id")
        return n
    if raw <= 0:
        raise HTTPException(status_code=400, detail="無效的 system_id")
    return int(raw)


def _put_english_system_for_exam_for_key(
    key: str, body: PutEnglishSystemForExamRequest
) -> EnglishSystemForExamSettingResponse:
    parsed = _parse_english_system_id_for_put(body.system_id)
    try:
        supabase = get_supabase()
        if parsed is None:
            row = _upsert_setting_and_get_row(supabase, key, "")
            if not row:
                return EnglishSystemForExamSettingResponse(key=key, system_id=None)
            return EnglishSystemForExamSettingResponse(
                key=key,
                system_id=None,
                system_setting_id=row.get("system_setting_id"),
                updated_at=to_taipei_iso(row.get("updated_at")),
                created_at=to_taipei_iso(row.get("created_at")),
            )
        value_to_save = str(parsed)
        chk = (
            supabase.table("English_System")
            .select("system_id")
            .eq("system_id", parsed)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if not chk.data or len(chk.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="找不到該 system_id 的 English_System 資料，或已刪除",
            )
        row = _upsert_setting_and_get_row(supabase, key, value_to_save)
        if not row:
            return EnglishSystemForExamSettingResponse(key=key, system_id=parsed)
        return EnglishSystemForExamSettingResponse(
            key=key,
            system_id=parsed,
            system_setting_id=row.get("system_setting_id"),
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/english-system-for-exam-localhost",
    response_model=EnglishSystemForExamSettingResponse,
)
def get_english_system_for_exam_localhost(_person_id: PersonId):
    """讀取 key=english_system_localhost 的供測驗 English_System system_id（value 轉 int）。無資料則為 null。"""
    try:
        supabase = get_supabase()
        return _get_english_system_for_exam_setting_row(
            supabase, ENGLISH_SYSTEM_EXAM_SETTING_KEY_LOCALHOST
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/english-system-for-exam-deploy",
    response_model=EnglishSystemForExamSettingResponse,
)
def get_english_system_for_exam_deploy(_person_id: PersonId):
    """讀取 key=english_system_deploy 的供測驗 English_System system_id（value 轉 int）。無資料則為 null。"""
    try:
        supabase = get_supabase()
        return _get_english_system_for_exam_setting_row(
            supabase, ENGLISH_SYSTEM_EXAM_SETTING_KEY_DEPLOY
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/english-system-for-exam-localhost",
    response_model=EnglishSystemForExamSettingResponse,
    summary="Put English System For Exam Localhost",
)
def put_english_system_for_exam_localhost(
    body: PutEnglishSystemForExamRequest, _person_id: PersonId
):
    """
    寫入供測驗 English_System（System_Setting key=english_system_localhost，value=str(system_id)）。
    該 key 已存在則更新，否則新增。正整數時需存在 English_System 且 deleted=false。
    system_id 傳空字串則將 value 清空（不檢查資料表）。
    """
    return _put_english_system_for_exam_for_key(
        ENGLISH_SYSTEM_EXAM_SETTING_KEY_LOCALHOST, body
    )


@router.put(
    "/english-system-for-exam-deploy",
    response_model=EnglishSystemForExamSettingResponse,
    summary="Put English System For Exam Deploy",
)
def put_english_system_for_exam_deploy(
    body: PutEnglishSystemForExamRequest, _person_id: PersonId
):
    """
    寫入供測驗 English_System（System_Setting key=english_system_deploy，value=str(system_id)）。
    行為同 english-system-for-exam-localhost，僅 key 不同。
    """
    return _put_english_system_for_exam_for_key(
        ENGLISH_SYSTEM_EXAM_SETTING_KEY_DEPLOY, body
    )
