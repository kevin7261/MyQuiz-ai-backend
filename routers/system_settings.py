"""
系統設定相關 API 模組。
- GET /system-settings/course-name：取得 course_name（key=course_name）。
- PUT /system-settings/course-name：寫入 course_name。

LLM／Deepgram API Key 請以環境變數設定（LLM_API_KEY 或 OPENAI_API_KEY、DEEPGRAM_API_KEY），不再經本表。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from utils.datetime_utils import now_taipei_iso, to_taipei_iso
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/system-settings", tags=["system-settings"])

SYSTEM_SETTING_COURSE_NAME_KEY = "course_name"
SYSTEM_SETTING_COLUMNS = "system_setting_id, key, value, updated_at, created_at"


class CourseNameResponse(BaseModel):
    """GET/PUT /system-settings/course-name 回應。"""
    system_setting_id: Optional[int] = None
    course_name: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class PutCourseNameRequest(BaseModel):
    """PUT /system-settings/course-name 的 body。"""
    course_name: str = Field(..., description="課程名稱")


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
            updated_at=to_taipei_iso(row.get("updated_at")),
            created_at=to_taipei_iso(row.get("created_at")),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
