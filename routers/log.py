"""
Log 表查詢 API。
- GET /log/logs：必填 query person_id、course_id；依 course_id 篩選，log_id 降冪。
  此端點不寫入 Log 表（避免查詢紀錄時產生遞迴 log）。
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from utils.supabase import get_supabase

router = APIRouter(tags=["log"])

LOG_COLUMNS = "log_id, person_id, course_id, api, api_metadata, updated_at, created_at"
_FETCH_PAGE = 1000


class LogRow(BaseModel):
    """對應 public.Log 一列。"""

    log_id: int
    person_id: str
    course_id: int = 0
    api: Optional[str] = None
    api_metadata: Optional[dict[str, Any]] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class LogListResponse(BaseModel):
    """GET /log/logs 回應。"""

    logs: list[LogRow] = Field(default_factory=list)
    count: int = Field(..., description="本次回傳筆數")


def _fetch_logs_by_course_id(course_id: int) -> list[dict]:
    """分批讀取指定 course_id 的 Log（PostgREST 單次有列數上限）。"""
    supabase = get_supabase()
    out: list[dict] = []
    offset = 0
    while True:
        try:
            resp = (
                supabase.table("Log")
                .select(LOG_COLUMNS)
                .eq("course_id", course_id)
                .order("log_id", desc=True)
                .range(offset, offset + _FETCH_PAGE - 1)
                .execute()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"讀取 Log 失敗：{e!s}") from e
        batch = resp.data or []
        if not batch:
            break
        out.extend(batch)
        if len(batch) < _FETCH_PAGE:
            break
        offset += _FETCH_PAGE
    return out


@router.get("/logs", response_model=LogListResponse)
def list_logs(_person_id: PersonId, course_id: CourseId) -> LogListResponse:
    """讀取指定 course_id 的 Log，依 log_id 降冪。person_id 僅供請求紀錄。"""
    rows = _fetch_logs_by_course_id(course_id)
    out: list[LogRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            out.append(LogRow.model_validate(r))
        except Exception:
            continue
    return LogListResponse(logs=out, count=len(out))
