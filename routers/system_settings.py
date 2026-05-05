"""
系統設定相關 API 模組。
- GET /system-settings/course-name：取得 course_name（key=course_name）。
- PUT /system-settings/course-name：寫入 course_name。
- GET /system-settings/person_analysis_user_prompt_text：取得 person_analysis_user_prompt_text（key=person_analysis_user_prompt_text）；須為有效登入使用者（不限 user_type）。
- PUT /system-settings/person_analysis_user_prompt_text：寫入 person_analysis_user_prompt_text；僅 user_type 1／2。

LLM／Deepgram API Key 請以環境變數設定（LLM_API_KEY 或 OPENAI_API_KEY、DEEPGRAM_API_KEY），不再經本表。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from utils.datetime_utils import now_taipei_iso
from utils.db_tables import USER_TABLE
from utils.supabase_client import get_supabase

# 與 routers/users.py 一致：deleted=false 或 null 為有效帳號
ACTIVE_USER_DELETED_FILTER = "deleted.eq.false,deleted.is.null"

router = APIRouter(prefix="/system-settings", tags=["system-settings"])

SYSTEM_SETTING_COURSE_NAME_KEY = "course_name"
SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY = "person_analysis_user_prompt_text"
SYSTEM_SETTING_COLUMNS = "system_setting_id, key, value"


def _user_type_for_active_person(person_id: str) -> Optional[int]:
    """依 person_id 查 User.user_type；無列或非有效帳號時回傳 None。"""
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .or_(ACTIVE_USER_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        ut = resp.data[0].get("user_type")
        if ut is None:
            return None
        return int(ut)
    except Exception:
        return None


def _require_active_person(person_id: str) -> None:
    """person_id 須對應有效 User（未刪除）。"""
    if _user_type_for_active_person(person_id) is None:
        raise HTTPException(status_code=404, detail="找不到該使用者")


def _require_developer_or_manager_for_person_analysis_prompt_write(person_id: str) -> None:
    """變更作答弱點分析報告規則：僅開發者（1）或管理者（2）。"""
    ut = _user_type_for_active_person(person_id)
    if ut is None:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    if ut not in (1, 2):
        raise HTTPException(status_code=403, detail="僅開發者或管理者可變更分析報告規則")


class CourseNameResponse(BaseModel):
    """GET/PUT /system-settings/course-name 回應。"""
    system_setting_id: Optional[int] = None
    course_name: Optional[str] = None


class PutCourseNameRequest(BaseModel):
    """PUT /system-settings/course-name 的 body。"""
    course_name: str = Field(..., description="課程名稱")


class PersonAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /system-settings/person_analysis_user_prompt_text 回應。"""
    system_setting_id: Optional[int] = None
    person_analysis_user_prompt_text: Optional[str] = None


class PutPersonAnalysisUserPromptTextRequest(BaseModel):
    """PUT /system-settings/person_analysis_user_prompt_text 的 body。"""
    person_analysis_user_prompt_text: str = Field(..., description="個人分析使用者 Prompt 文字")


@router.get("/course-name", response_model=CourseNameResponse)
def get_course_name_setting(_person_id: PersonId):
    """取得 course_name（System_Setting key=course_name）。"""
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
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/person_analysis_user_prompt_text", response_model=PersonAnalysisUserPromptTextResponse)
def get_person_analysis_user_prompt_text_setting(person_id: PersonId):
    """取得 person_analysis_user_prompt_text（System_Setting key=person_analysis_user_prompt_text）。"""
    _require_active_person(person_id)
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return PersonAnalysisUserPromptTextResponse()
        row = resp.data[0]
        return PersonAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            person_analysis_user_prompt_text=row.get("value"),
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
    """寫入 course_name（System_Setting key=course_name）。"""
    value_to_save = (body.course_name or "").strip() or ""
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(supabase, SYSTEM_SETTING_COURSE_NAME_KEY, value_to_save)
        if not row:
            return CourseNameResponse(course_name=value_to_save or None)
        return CourseNameResponse(
            system_setting_id=row.get("system_setting_id"),
            course_name=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/person_analysis_user_prompt_text", response_model=PersonAnalysisUserPromptTextResponse)
def put_person_analysis_user_prompt_text_setting(
    body: PutPersonAnalysisUserPromptTextRequest, person_id: PersonId
):
    """寫入 person_analysis_user_prompt_text（System_Setting key=person_analysis_user_prompt_text）。"""
    _require_developer_or_manager_for_person_analysis_prompt_write(person_id)
    value_to_save = (body.person_analysis_user_prompt_text or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase, SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY, value_to_save
        )
        if not row:
            return PersonAnalysisUserPromptTextResponse(
                person_analysis_user_prompt_text=value_to_save or None
            )
        return PersonAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            person_analysis_user_prompt_text=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
