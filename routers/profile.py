"""
使用者相關 API 模組。
提供：
- GET /users：列出指定學院（必填 query college_id）的 User 表（含 password），含各使用者選課 courses 列表（唯讀）
- PUT /users/me/password：修改呼叫者（token 解析）自己的密碼
- POST /auth/login：以 person_id + password 登入；成功時簽發 access_token（Bearer）
  並回傳該帳號之 User_Course_Relation 課程列表
- POST /auth/refresh：以仍有效的 token 換發新 token（延長效期）

新增／編輯／刪除使用者請用 /v1/rag/course-members。
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from dependencies.person_id import CurrentUser
from pydantic import BaseModel, Field

from utils.auth import issue_token, token_ttl_seconds
from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.db_schema import (
    ACTIVE_DELETED_FILTER,
    COLLEGE_TABLE,
    COURSE_TABLE,
    USER_COURSE_RELATION_TABLE,
    USER_TABLE,
)
from utils.openapi import openapi_body
from utils.supabase import get_supabase

router = APIRouter(tags=["profile"])

_logger = logging.getLogger(__name__)

# User 表實體欄位（user_type 隨 User_Course_Relation 各列；college_name 可自 College 串接）
USER_TABLE_COLUMNS = "user_id, person_id, college_id, college_name, name, deleted, updated_at, created_at"

# 管理者 / 教師身分（可列出該學院使用者）
COLLEGE_MANAGER_USER_TYPES = (1, 2)


def _normalize_college_id(raw: Any) -> str:
    return str(raw or "").strip()


def _caller_is_college_manager(supabase, person_id: str, college_id: Any) -> bool:
    """呼叫者在該學院是否具管理者（user_type 1）或教師（2）身分。

    依 User_Course_Relation（該學院任一有效選課列）判定；查詢失敗一律視為無權限（fail-closed）。
    """
    pid = (person_id or "").strip()
    cid = _normalize_college_id(college_id)
    if not pid or not cid:
        return False
    try:
        cid_int = int(cid)
    except ValueError:
        return False
    try:
        resp = (
            supabase.table(USER_COURSE_RELATION_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .eq("college_id", cid_int)
            .or_(ACTIVE_DELETED_FILTER)
            .execute()
        )
    except Exception:
        _logger.exception("查詢呼叫者學院角色失敗 person_id=%s college_id=%s", pid, cid)
        return False
    return any(
        int(r.get("user_type") or 0) in COLLEGE_MANAGER_USER_TYPES
        for r in (resp.data or [])
    )


def _scope_user_query_by_college(query, college_id: Any):
    """身分 = person_id + college_id：token 帶 college_id 時補上學校範圍，
    確保跨校同 person_id 只命中該學校那一列。舊版無 cid 的 token 不加範圍（請重新登入取得新 token）。"""
    cid = _normalize_college_id(college_id)
    return query.eq("college_id", cid) if cid else query


def _fetch_colleges_by_ids(supabase, college_ids: list[str]) -> dict[str, str]:
    """依 college_id 自 College 表批次查詢 college_name。"""
    ids: list[int] = []
    for cid in college_ids:
        c = _normalize_college_id(cid)
        if not c:
            continue
        try:
            ids.append(int(c))
        except ValueError:
            continue
    if not ids:
        return {}
    resp = (
        supabase.table(COLLEGE_TABLE)
        .select("college_id, college_name")
        .in_("college_id", list(dict.fromkeys(ids)))
        .or_(ACTIVE_DELETED_FILTER)
        .execute()
    )
    out: dict[str, str] = {}
    for row in resp.data or []:
        cid = row.get("college_id")
        if cid is None:
            continue
        out[str(cid)] = (row.get("college_name") or "").strip()
    return out


def _college_name_for_user(user_row: dict, college_by_id: dict[str, str] | None = None) -> Optional[str]:
    cid = _normalize_college_id(user_row.get("college_id"))
    stored = (user_row.get("college_name") or "").strip()
    if cid and college_by_id:
        joined = (college_by_id.get(cid) or "").strip()
        if joined:
            return joined
    return stored or None


def _fetch_courses_by_ids(supabase, course_ids: list[int]) -> dict[int, dict]:
    """依 course_id 自 Course 表批次查詢課程資訊。"""
    ids = [i for i in dict.fromkeys(course_ids) if i]
    if not ids:
        return {}
    resp = (
        supabase.table(COURSE_TABLE)
        .select("course_id, college_id, semester, course_name")
        .in_("course_id", ids)
        .or_(ACTIVE_DELETED_FILTER)
        .execute()
    )
    out: dict[int, dict] = {}
    for row in resp.data or []:
        cid = row.get("course_id")
        if cid is not None:
            out[int(cid)] = row
    return out


def _fetch_course_relations_by_user_ids(supabase, user_ids: list[int]) -> dict[int, list[dict]]:
    """各 user_id 於 User_Course_Relation 之全部有效列（依 course_user_id 排序）。"""
    if not user_ids:
        return {}
    resp = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id, user_id, course_id, college_id, user_type")
        .in_("user_id", user_ids)
        .or_(ACTIVE_DELETED_FILTER)
        .order("course_user_id")
        .execute()
    )
    out: dict[int, list[dict]] = {}
    for row in resp.data or []:
        uid = row.get("user_id")
        if uid is None:
            continue
        out.setdefault(int(uid), []).append(row)
    return out


def _build_user_course_items(
    relations: list[dict],
    course_by_id: dict[int, dict] | None = None,
) -> list[dict]:
    """將 User_Course_Relation 列轉為 API 課程項目（course_name 自 Course 表串接）。"""
    course_by_id = course_by_id or {}
    items: list[dict] = []
    for r in relations:
        course_user_id = r.get("course_user_id")
        if course_user_id is None:
            continue
        course_id = int(r.get("course_id") or 0)
        course_row = course_by_id.get(course_id, {})
        college_raw = r.get("college_id")
        if college_raw is None or int(college_raw or 0) == 0:
            college_raw = course_row.get("college_id")
        college_id = int(college_raw) if college_raw is not None and int(college_raw or 0) != 0 else None
        items.append({
            "course_user_id": int(course_user_id),
            "course_id": course_id,
            "college_id": college_id,
            "course_name": (course_row.get("course_name") or "").strip() or None,
            "semester": (course_row.get("semester") or "").strip() or None,
            "user_type": r.get("user_type"),
        })
    return items


def _user_public_dict(
    user_row: dict,
    college_by_id: dict[str, str] | None = None,
    courses: list[dict] | None = None,
    *,
    include_password: bool = False,
) -> dict:
    """組出對外使用者 dict；user_type 見 courses 各項。"""
    cid = _normalize_college_id(user_row.get("college_id"))
    out = {k: user_row.get(k) for k in ("user_id", "person_id", "name", "updated_at", "created_at")}
    out["college_id"] = cid or None
    out["college_name"] = _college_name_for_user(user_row, college_by_id)
    out["courses"] = courses if courses is not None else []
    out["user_metadata"] = None
    if include_password:
        out["password"] = user_row.get("password")
    out["updated_at"] = to_taipei_iso(out.get("updated_at"))
    out["created_at"] = to_taipei_iso(out.get("created_at"))
    return out


class UserListItem(BaseModel):
    """單筆使用者；user_type 見 courses 各項。password 僅 GET /users 回傳。"""
    user_id: int
    person_id: Optional[str] = None
    college_id: Optional[str] = None
    college_name: Optional[str] = None
    name: Optional[str] = None
    password: Optional[str] = None
    courses: list["UserCourseItem"] = Field(default_factory=list)
    user_metadata: Optional[Any] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class ListUsersResponse(BaseModel):
    """GET /users 回應。"""
    users: list[UserListItem]
    count: int


class LoginRequest(BaseModel):
    """POST /auth/login 請求：person_id + password + college_id（皆必填）。
    college_id 為學校（學院）id；後端會驗證該帳號確實屬於此學校，不符回 403。"""
    person_id: str
    password: str
    college_id: int = Field(..., description="必填；學校（學院）id，登入時驗證帳號屬於此學校")


class UserCourseItem(BaseModel):
    """User_Course_Relation 單筆課程／選課資訊（含 user_type；course_name / semester 自 Course 表串接）。"""
    course_user_id: int
    course_id: int
    college_id: Optional[int] = None
    course_name: Optional[str] = None
    semester: Optional[str] = None
    user_type: Optional[int] = None


UserListItem.model_rebuild()


class LoginResponse(BaseModel):
    """登入成功回傳使用者資訊（不含 password）；user.courses 與頂層 courses 皆為該帳號選課列。
    access_token 供之後所有 API 以 `Authorization: Bearer <access_token>` 帶入。"""
    user: UserListItem
    courses: list[UserCourseItem] = Field(default_factory=list)
    access_token: str = Field(..., description="Bearer token；之後請求帶 Authorization header")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="access_token 效期（秒）；預設 30 天，可由 env AUTH_TOKEN_TTL_SECONDS 調整")


def _courses_for_users(
    supabase,
    user_ids: list[int],
) -> dict[int, list[UserCourseItem]]:
    """批次取得各 user 的選課列表。"""
    rel_by_uid = _fetch_course_relations_by_user_ids(supabase, user_ids)
    all_course_ids: list[int] = []
    for rels in rel_by_uid.values():
        for r in rels:
            cid = int(r.get("course_id") or 0)
            if cid:
                all_course_ids.append(cid)
    course_by_id = _fetch_courses_by_ids(supabase, all_course_ids)
    out: dict[int, list[UserCourseItem]] = {}
    for uid in user_ids:
        items = _build_user_course_items(rel_by_uid.get(uid, []), course_by_id)
        out[uid] = [UserCourseItem(**item) for item in items]
    return out


def _user_list_item(
    user_row: dict,
    college_by_id: dict[str, str] | None,
    courses: list[UserCourseItem] | None = None,
    *,
    include_password: bool = False,
) -> UserListItem:
    course_dicts = [c.model_dump() for c in (courses or [])]
    return UserListItem(
        **_user_public_dict(user_row, college_by_id, course_dicts, include_password=include_password)
    )


@router.get("/users", response_model=ListUsersResponse)
def list_users(
    caller: CurrentUser,
    college_id: int = Query(..., description="必填；僅列出該學院（User.college_id）的使用者"),
):
    """
    列出指定學院（必填 query `college_id`）的 User 表內容；僅 deleted = false。

    權限：僅該學院的管理者（user_type 1）或教師（2）可呼叫，且 `college_id` 須等於
    登入身分所屬學校（token 的 college_id），否則 403。
    回應**不含 password**（密碼不再對外回傳）。唯讀；新增／編輯／刪除請用 /v1/bank/course-members。
    """
    # 身分 = person_id + college_id：只能查自己所屬學校，跨校查詢一律拒絕。
    if _normalize_college_id(college_id) != _normalize_college_id(caller.college_id):
        raise HTTPException(
            status_code=403, detail="僅能查詢自己所屬學校（college_id）的使用者"
        )
    supabase = get_supabase()
    if not _caller_is_college_manager(supabase, caller.person_id, caller.college_id):
        raise HTTPException(
            status_code=403, detail="僅該學院的管理者或教師可列出使用者"
        )
    try:
        resp = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("deleted", False)
            .eq("college_id", college_id)
            .execute()
        )
        rows = resp.data or []
        uids = [r["user_id"] for r in rows if r.get("user_id") is not None]
        courses_by_uid = _courses_for_users(supabase, uids)
        college_map = _fetch_colleges_by_ids(
            supabase,
            [_normalize_college_id(r.get("college_id")) for r in rows],
        )
        return ListUsersResponse(
            users=[
                _user_list_item(
                    r,
                    college_map,
                    courses_by_uid.get(r["user_id"], []),
                )
                for r in rows
            ],
            count=len(rows),
        )
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "nodename" in err or "errno 8" in err or "name or service not known" in err:
            raise HTTPException(
                status_code=503,
                detail="無法連線至 Supabase，請確認 .env 的 SUPABASE_URL 正確且網路可連線。",
            )
        _logger.exception("GET /users 失敗 college_id=%s", college_id)
        raise HTTPException(status_code=500, detail="列出使用者失敗，請稍後再試")


class ProfileResponse(BaseModel):
    """GET /users/me 回應：呼叫者自己的 profile（不含 password）。"""
    user_id: int
    person_id: Optional[str] = None
    college_id: Optional[str] = None
    college_name: Optional[str] = None
    name: Optional[str] = None
    courses: list[UserCourseItem] = Field(default_factory=list)
    user_metadata: Optional[Any] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


@router.get("/users/me", response_model=ProfileResponse)
def get_my_profile(caller: CurrentUser):
    """
    回傳呼叫者自己的 profile（呼叫者由 Authorization token 解析），不含 password。
    身分為 person_id + college_id；跨校同 person_id 時僅回傳登入學校那一筆。
    欄位與 login 回傳的 user 相同；登入後若需重新取得最新使用者／選課資料時使用。
    """
    person_id = caller.person_id
    try:
        supabase = get_supabase()
        query = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("person_id", person_id)
        )
        query = _scope_user_query_by_college(query, caller.college_id)
        resp = query.or_(ACTIVE_DELETED_FILTER).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
        row = resp.data[0]
        uid = row.get("user_id")
        courses = _courses_for_users(supabase, [int(uid)]).get(int(uid), []) if uid is not None else []
        college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(row.get("college_id"))])
        pub = _user_public_dict(row, college_map, [c.model_dump() for c in courses])
        return ProfileResponse(**pub)
    except HTTPException:
        raise
    except Exception:
        _logger.exception("GET /users/me 失敗 person_id=%s", person_id)
        raise HTTPException(status_code=500, detail="取得使用者資料失敗，請稍後再試")


class UpdateMyPasswordRequest(BaseModel):
    """PUT /users/me/password 請求；呼叫者由 token 解析。"""
    password: str = Field(..., min_length=1, description="新密碼")


class UpdateMyPasswordResponse(BaseModel):
    """PUT /users/me/password 回應。"""
    message: str
    person_id: str
    updated_at: Optional[str] = None


class RefreshTokenResponse(BaseModel):
    """POST /auth/refresh 回應。"""
    access_token: str = Field(..., description="新簽發的 Bearer token")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="access_token 效期（秒）")


@router.put("/users/me/password", response_model=UpdateMyPasswordResponse)
def update_my_password(
    body: openapi_body(UpdateMyPasswordRequest, {"password": "string"}),
    caller: CurrentUser,
):
    """
    修改呼叫者自己的密碼（呼叫者由 Authorization token 解析，body 只需新密碼）。
    身分為 person_id + college_id；僅更新登入學校那一筆，不會動到跨校同 person_id 的其他帳號。
    """
    person_id = caller.person_id
    pwd = (body.password or "").strip()
    if not pwd:
        raise HTTPException(status_code=400, detail="請傳入 password")
    try:
        supabase = get_supabase()
        ts = now_taipei_iso()
        query = (
            supabase.table(USER_TABLE)
            .update({"password": pwd, "updated_at": ts})
            .eq("person_id", person_id)
        )
        query = _scope_user_query_by_college(query, caller.college_id)
        resp = query.or_(ACTIVE_DELETED_FILTER).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
        return UpdateMyPasswordResponse(
            message="密碼已更新",
            person_id=person_id,
            updated_at=to_taipei_iso(resp.data[0].get("updated_at")) or ts,
        )
    except HTTPException:
        raise
    except Exception:
        _logger.exception("PUT /users/me/password 失敗 person_id=%s", person_id)
        raise HTTPException(status_code=500, detail="更新密碼失敗，請稍後再試")


@router.post("/auth/refresh", response_model=RefreshTokenResponse)
def refresh_access_token(caller: CurrentUser):
    """
    以仍有效的 token 換發新 token（延長效期）；token 已過期會回 401，需重新登入。
    新 token 沿用原 token 的 person_id 與 college_id（身分情境不變）。
    前端可於 token 接近到期（見 expires_in）時呼叫，避免使用中途被登出。
    """
    return RefreshTokenResponse(
        access_token=issue_token(caller.person_id, caller.college_id),
        expires_in=token_ttl_seconds(),
    )


@router.post("/auth/login", response_model=LoginResponse)
def login(
    body: openapi_body(
        LoginRequest,
        {"person_id": "string", "password": "string", "college_id": 0},
    ),
):
    """
    以 person_id、password 與 college_id（學校id）驗證登入（唯一不需要 token 的端點）；三者皆必填。
    後端會驗證該帳號屬於 college_id 此學校，不符回 403。
    成功回傳使用者資訊（不含 password）、課程列表與 access_token；
    之後所有 API 請帶 `Authorization: Bearer <access_token>`。
    """
    person_id = (body.person_id or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id")
    pwd = (body.password or "").strip()
    cols = f"{USER_TABLE_COLUMNS}, password"
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(cols)
            .eq("person_id", person_id)
            .or_(ACTIVE_DELETED_FILTER)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        # 身分 = person_id + college_id：同 person_id 可能跨多校多筆，先挑出指定學校那一筆再驗密碼，
        # 不可只取 rows[0]（否則第一筆非該校時會誤判密碼／學校）。
        target_cid = _normalize_college_id(body.college_id)
        row = next(
            (r for r in rows if _normalize_college_id(r.get("college_id")) == target_cid),
            None,
        )
        if row is None:
            raise HTTPException(status_code=403, detail="此帳號不屬於指定的學校（college_id）")
        if (row.get("password") or "").strip() != pwd:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        uid = row.get("user_id")
        courses = _courses_for_users(supabase, [int(uid)]).get(int(uid), []) if uid is not None else []
        college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(row.get("college_id"))])
        user = _user_list_item(row, college_map, courses)
        return LoginResponse(
            user=user,
            courses=courses,
            access_token=issue_token(person_id, _normalize_college_id(row.get("college_id"))),
            expires_in=token_ttl_seconds(),
        )
    except HTTPException:
        raise
    except Exception:
        _logger.exception("POST /auth/login 失敗 person_id=%s", person_id)
        raise HTTPException(status_code=500, detail="登入失敗，請稍後再試")
