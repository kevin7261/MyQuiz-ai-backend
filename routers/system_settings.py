"""
系統設定相關 API 模組。
- GET /system-settings/person_analysis_user_prompt_text：取得 person_analysis_user_prompt_text（key=person_analysis_user_prompt_text）；須為有效登入使用者（不限 user_type）；必填 query `course_id`。
- PUT /system-settings/person_analysis_user_prompt_text：寫入 person_analysis_user_prompt_text；僅 user_type 1／2；必填 query `course_id`。
- GET /system-settings/course_analysis_user_prompt_text：取得 course_analysis_user_prompt_text（key=course_analysis_user_prompt_text）；須為有效登入使用者（不限 user_type）；必填 query `course_id`。
- PUT /system-settings/course_analysis_user_prompt_text：寫入 course_analysis_user_prompt_text；僅 user_type 1／2；必填 query `course_id`。

LLM API Key 請以環境變數設定（LLM_API_KEY 或 OPENAI_API_KEY），不再經本表。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from utils.taipei_time import now_taipei_iso
from utils.db_schema import ACTIVE_DELETED_FILTER, USER_COURSE_RELATION_TABLE, USER_TABLE
from utils.openapi import openapi_body
from utils.supabase import get_supabase

router = APIRouter(prefix="/system-settings", tags=["system-settings"])

SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY = "person_analysis_user_prompt_text"
SYSTEM_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY = "course_analysis_user_prompt_text"
SYSTEM_SETTING_COLUMNS = "system_setting_id, course_id, key, value"


def fetch_system_setting_text(key: str, course_id: int) -> str:
    """讀取 System_Setting value（依 course_id + key）；失敗或無列時回傳空字串。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select("value")
            .eq("key", key)
            .eq("course_id", course_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return ""
        return (resp.data[0].get("value") or "").strip()
    except Exception:
        return ""


def _user_type_for_active_person(person_id: str, course_id: int) -> Optional[int]:
    """依 person_id、course_id 查 User_Course_Relation.user_type；無列或非有效帳號時回傳 None。"""
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_COURSE_RELATION_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .or_(ACTIVE_DELETED_FILTER)
            .order("course_user_id")
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
    pid = (person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select("user_id")
            .eq("person_id", pid)
            .or_(ACTIVE_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="找不到該使用者")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="找不到該使用者")


def _require_developer_or_manager_for_analysis_prompt_write(
    person_id: str, course_id: int
) -> None:
    """變更個人／課程分析報告規則：僅開發者（1）或管理者（2）。"""
    ut = _user_type_for_active_person(person_id, course_id)
    if ut is None:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    if ut not in (1, 2):
        raise HTTPException(status_code=403, detail="僅開發者或管理者可變更分析報告規則")


class PersonAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /system-settings/person_analysis_user_prompt_text 回應。"""
    system_setting_id: Optional[int] = None
    course_id: Optional[int] = None
    person_analysis_user_prompt_text: Optional[str] = None


class PutPersonAnalysisUserPromptTextRequest(BaseModel):
    """PUT /system-settings/person_analysis_user_prompt_text 的 body。"""
    person_analysis_user_prompt_text: str = Field(..., description="個人分析使用者 Prompt 文字")


class CourseAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /system-settings/course_analysis_user_prompt_text 回應。"""
    system_setting_id: Optional[int] = None
    course_id: Optional[int] = None
    course_analysis_user_prompt_text: Optional[str] = None


class PutCourseAnalysisUserPromptTextRequest(BaseModel):
    """PUT /system-settings/course_analysis_user_prompt_text 的 body。"""
    course_analysis_user_prompt_text: str = Field(..., description="課程分析使用者 Prompt 文字")


@router.get("/person_analysis_user_prompt_text", response_model=PersonAnalysisUserPromptTextResponse)
def get_person_analysis_user_prompt_text_setting(person_id: PersonId, course_id: CourseId):
    """取得 person_analysis_user_prompt_text（System_Setting key=person_analysis_user_prompt_text）。"""
    _require_active_person(person_id)
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY)
            .eq("course_id", course_id)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return PersonAnalysisUserPromptTextResponse(course_id=course_id)
        row = resp.data[0]
        return PersonAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            course_id=row.get("course_id"),
            person_analysis_user_prompt_text=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _upsert_setting_and_get_row(supabase, key: str, value: str, course_id: int):
    """依 course_id + key 新增或更新一筆 System_Setting，回傳該筆 row（dict）。"""
    now = now_taipei_iso()
    resp = (
        supabase.table("System_Setting")
        .select("system_setting_id")
        .eq("key", key)
        .eq("course_id", course_id)
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
            "course_id": course_id,
            "key": key,
            "value": value,
            "updated_at": now,
            "created_at": now,
        }).execute()
    resp2 = (
        supabase.table("System_Setting")
        .select(SYSTEM_SETTING_COLUMNS)
        .eq("key", key)
        .eq("course_id", course_id)
        .limit(1)
        .execute()
    )
    if not resp2.data or len(resp2.data) == 0:
        return None
    return resp2.data[0]


@router.put("/person_analysis_user_prompt_text", response_model=PersonAnalysisUserPromptTextResponse)
def put_person_analysis_user_prompt_text_setting(
    body: openapi_body(
        PutPersonAnalysisUserPromptTextRequest,
        {"person_analysis_user_prompt_text": "string"},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 person_analysis_user_prompt_text（System_Setting key=person_analysis_user_prompt_text）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.person_analysis_user_prompt_text or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            SYSTEM_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            return PersonAnalysisUserPromptTextResponse(
                course_id=course_id,
                person_analysis_user_prompt_text=value_to_save or None,
            )
        return PersonAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            course_id=row.get("course_id"),
            person_analysis_user_prompt_text=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/course_analysis_user_prompt_text", response_model=CourseAnalysisUserPromptTextResponse)
def get_course_analysis_user_prompt_text_setting(person_id: PersonId, course_id: CourseId):
    """取得 course_analysis_user_prompt_text（System_Setting key=course_analysis_user_prompt_text）。"""
    _require_active_person(person_id)
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("System_Setting")
            .select(SYSTEM_SETTING_COLUMNS)
            .eq("key", SYSTEM_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY)
            .eq("course_id", course_id)
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return CourseAnalysisUserPromptTextResponse(course_id=course_id)
        row = resp.data[0]
        return CourseAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            course_id=row.get("course_id"),
            course_analysis_user_prompt_text=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/course_analysis_user_prompt_text", response_model=CourseAnalysisUserPromptTextResponse)
def put_course_analysis_user_prompt_text_setting(
    body: openapi_body(
        PutCourseAnalysisUserPromptTextRequest,
        {"course_analysis_user_prompt_text": "string"},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入 course_analysis_user_prompt_text（System_Setting key=course_analysis_user_prompt_text）。"""
    _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
    value_to_save = (body.course_analysis_user_prompt_text or "").strip()
    try:
        supabase = get_supabase()
        row = _upsert_setting_and_get_row(
            supabase,
            SYSTEM_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            return CourseAnalysisUserPromptTextResponse(
                course_id=course_id,
                course_analysis_user_prompt_text=value_to_save or None,
            )
        return CourseAnalysisUserPromptTextResponse(
            system_setting_id=row.get("system_setting_id"),
            course_id=row.get("course_id"),
            course_analysis_user_prompt_text=row.get("value"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
