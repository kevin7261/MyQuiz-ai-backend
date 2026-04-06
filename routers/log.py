"""
Log 表查詢 API。
- GET /log/logs：必填 query person_id（供 APILogMiddleware 寫入 Log，不用於篩選）；回傳整張 Log，依 log_id 降冪。
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/log", tags=["log"])

LOG_COLUMNS = "log_id, person_id, api, api_metadata, updated_at, created_at"
_FETCH_PAGE = 1000


class LogRow(BaseModel):
    """對應 public.Log 一列。"""

    log_id: int
    person_id: str
    api: Optional[str] = None
    api_metadata: Optional[dict[str, Any]] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class LogListResponse(BaseModel):
    """GET /log/logs 回應。"""

    logs: list[LogRow] = Field(default_factory=list)
    count: int = Field(..., description="本次回傳筆數")


def _fetch_all_logs() -> list[dict]:
    """分批讀取 Log（PostgREST 單次有列數上限）。"""
    supabase = get_supabase()
    out: list[dict] = []
    offset = 0
    while True:
        try:
            resp = (
                supabase.table("Log")
                .select(LOG_COLUMNS)
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
def list_logs(_person_id: PersonId) -> LogListResponse:
    """讀取整張 Log 表，依 log_id 降冪。person_id 僅供請求紀錄，不影響查詢結果。"""
    rows = _fetch_all_logs()
    out: list[LogRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            out.append(LogRow.model_validate(r))
        except Exception:
            continue
    return LogListResponse(logs=out, count=len(out))
