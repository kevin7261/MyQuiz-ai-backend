"""
課程（Course）相關 API 模組。
- GET /course/courses：列出 Course 表（僅 deleted = false 或 null），含所屬學院 college_id／college_name
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dependencies.person_id import PersonId
from utils.db_schema import ACTIVE_DELETED_FILTER, COLLEGE_TABLE, COURSE_TABLE
from utils.supabase import get_supabase
from utils.taipei_time import to_taipei_iso

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["course"])

COURSE_TABLE_COLUMNS = "course_id, college_id, semester, course_name, deleted, updated_at, created_at"
COLLEGE_TABLE_COLUMNS = "college_id, college_name"


class CourseListItem(BaseModel):
    """單筆課程（一門課對應一個學院）。"""
    course_id: int
    college_id: Optional[int] = None
    college_name: Optional[str] = None
    semester: Optional[str] = None
    course_name: Optional[str] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class ListCoursesResponse(BaseModel):
    """GET /course/courses 回應。"""
    courses: list[CourseListItem]
    count: int


def _fetch_colleges_by_ids(supabase, college_ids: list[int]) -> dict[int, dict]:
    ids = [i for i in dict.fromkeys(college_ids) if i]
    if not ids:
        return {}
    resp = (
        supabase.table(COLLEGE_TABLE)
        .select(COLLEGE_TABLE_COLUMNS)
        .in_("college_id", ids)
        .or_(ACTIVE_DELETED_FILTER)
        .execute()
    )
    out: dict[int, dict] = {}
    for row in resp.data or []:
        cid = row.get("college_id")
        if cid is not None:
            out[int(cid)] = row
    return out


def _course_public_dict(row: dict, college_by_id: dict[int, dict] | None = None) -> dict:
    college_id = row.get("college_id")
    college_id_int = int(college_id) if college_id is not None and int(college_id or 0) != 0 else None
    college_name: str | None = None
    if college_id_int and college_by_id:
        college_row = college_by_id.get(college_id_int)
        if college_row:
            college_name = (college_row.get("college_name") or "").strip() or None
    return {
        "course_id": row.get("course_id"),
        "college_id": college_id_int,
        "college_name": college_name,
        "semester": (row.get("semester") or "").strip() or None,
        "course_name": (row.get("course_name") or "").strip() or None,
        "updated_at": to_taipei_iso(row.get("updated_at")),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


@router.get("/courses", response_model=ListCoursesResponse)
def list_courses(_person_id: PersonId):
    """列出 Course 表內容；僅回傳未刪除之列，每門課附一個學院（college_id、college_name）。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COURSE_TABLE)
            .select(COURSE_TABLE_COLUMNS)
            .or_(ACTIVE_DELETED_FILTER)
            .order("course_id")
            .execute()
        )
        rows = resp.data or []
        college_ids = [
            int(r["college_id"])
            for r in rows
            if r.get("college_id") is not None and int(r.get("college_id") or 0) != 0
        ]
        college_by_id = _fetch_colleges_by_ids(supabase, college_ids)
        return ListCoursesResponse(
            courses=[CourseListItem(**_course_public_dict(r, college_by_id)) for r in rows],
            count=len(rows),
        )
    except Exception as e:
        err = str(e).lower()
        if "nodename" in err or "errno 8" in err or "name or service not known" in err:
            raise HTTPException(
                status_code=503,
                detail="無法連線至 Supabase，請確認 .env 的 SUPABASE_URL 正確且網路可連線。",
            )
        _logger.exception("GET /courses 失敗")
        raise HTTPException(status_code=500, detail="列出課程失敗，請稍後再試")
