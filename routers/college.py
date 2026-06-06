"""
學院（College）相關 API 模組。
- GET /college/colleges：列出 College 表（僅 deleted = false 或 null），含所屬課程
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from utils.db_schema import ACTIVE_DELETED_FILTER, COLLEGE_TABLE, COURSE_TABLE
from utils.supabase import get_supabase
from utils.taipei_time import to_taipei_iso

router = APIRouter(tags=["college"])

COLLEGE_TABLE_COLUMNS = "college_id, college_name, deleted, updated_at, created_at"
COURSE_TABLE_COLUMNS = "course_id, college_id, semester, course_name"


class CourseEmbed(BaseModel):
    """學院底下課程。"""
    course_id: int
    college_id: Optional[int] = None
    semester: Optional[str] = None
    course_name: Optional[str] = None


class CollegeListItem(BaseModel):
    """單筆學院。"""
    college_id: int
    college_name: Optional[str] = None
    courses: list[CourseEmbed] = Field(default_factory=list)
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class ListCollegesResponse(BaseModel):
    """GET /college/colleges 回應。"""
    colleges: list[CollegeListItem]
    count: int


def _fetch_courses_by_college_ids(supabase, college_ids: list[int]) -> dict[int, list[dict]]:
    ids = [i for i in dict.fromkeys(college_ids) if i]
    if not ids:
        return {}
    resp = (
        supabase.table(COURSE_TABLE)
        .select(COURSE_TABLE_COLUMNS)
        .in_("college_id", ids)
        .or_(ACTIVE_DELETED_FILTER)
        .order("course_id")
        .execute()
    )
    out: dict[int, list[dict]] = {}
    for row in resp.data or []:
        cid = row.get("college_id")
        if cid is None:
            continue
        out.setdefault(int(cid), []).append(row)
    return out


def _college_public_dict(row: dict, courses_by_college: dict[int, list[dict]] | None = None) -> dict:
    college_id = row.get("college_id")
    college_id_int = int(college_id) if college_id is not None else None
    course_rows = (courses_by_college or {}).get(college_id_int, []) if college_id_int else []
    courses = [
        CourseEmbed(
            course_id=int(c["course_id"]),
            college_id=int(c.get("college_id") or college_id_int) if c.get("college_id") is not None else college_id_int,
            semester=(c.get("semester") or "").strip() or None,
            course_name=(c.get("course_name") or "").strip() or None,
        )
        for c in course_rows
        if c.get("course_id") is not None
    ]
    return {
        "college_id": college_id,
        "college_name": (row.get("college_name") or "").strip() or None,
        "courses": courses,
        "updated_at": to_taipei_iso(row.get("updated_at")),
        "created_at": to_taipei_iso(row.get("created_at")),
    }


@router.get("/colleges", response_model=ListCollegesResponse)
def list_colleges(_person_id: PersonId):
    """列出 College 表內容；僅回傳未刪除之列，並附所屬課程。"""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(COLLEGE_TABLE)
            .select(COLLEGE_TABLE_COLUMNS)
            .or_(ACTIVE_DELETED_FILTER)
            .order("college_id")
            .execute()
        )
        rows = resp.data or []
        college_ids = [int(r["college_id"]) for r in rows if r.get("college_id") is not None]
        courses_by_college = _fetch_courses_by_college_ids(supabase, college_ids)
        return ListCollegesResponse(
            colleges=[CollegeListItem(**_college_public_dict(r, courses_by_college)) for r in rows],
            count=len(rows),
        )
    except Exception as e:
        err = str(e).lower()
        if "nodename" in err or "errno 8" in err or "name or service not known" in err:
            raise HTTPException(
                status_code=503,
                detail="無法連線至 Supabase，請確認 .env 的 SUPABASE_URL 正確且網路可連線。",
            )
        raise HTTPException(status_code=500, detail=str(e))
