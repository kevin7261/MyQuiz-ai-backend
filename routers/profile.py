"""
使用者相關 API 模組。
提供：
- GET /profile/users：列出 User 表（含 password），含各使用者選課 courses 列表
- POST /profile/users：新增單一使用者（person_id、name、course_id、user_type；college_id 自 Course 帶入）
- POST /profile/users/batch：批次新增使用者（每筆僅 person_id、name；user_type 固定為 3；password 預設 0000）
- PUT /profile/users/delete：軟刪除（body.person_id 指定對象，將 deleted 設為 true）
- POST /profile/login：以 person_id + password 登入（成功時另回傳該帳號之 User_Course_Relation 課程列表）
- PATCH /profile：更新個人資料（name、user_type）
"""

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, HTTPException

from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

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

router = APIRouter(prefix="/profile", tags=["profile"])

# User 表實體欄位（user_type 隨 User_Course_Relation 各列；college_name 可自 College 串接）
USER_TABLE_COLUMNS = "user_id, person_id, college_id, college_name, name, deleted, updated_at, created_at"


def _normalize_college_id(raw: Any) -> str:
    return str(raw or "").strip()


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


def _fetch_all_course_relations_for_user(supabase, user_id: int) -> list[dict]:
    return _fetch_course_relations_by_user_ids(supabase, [user_id]).get(user_id, [])


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


def _insert_user_course_relation(
    supabase,
    *,
    user_id: int,
    person_id: str,
    name: str,
    user_type: int,
    ts: str,
    college_id: int = 0,
    course_id: int = 0,
) -> None:
    supabase.table(USER_COURSE_RELATION_TABLE).insert({
        "user_id": user_id,
        "person_id": person_id,
        "name": (name or "").strip() or "",
        "course_id": course_id,
        "college_id": college_id,
        "user_type": user_type,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }).execute()


class UserListItem(BaseModel):
    """單筆使用者；user_type 見 courses 各項。password 僅 GET /profile/users 回傳。"""
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
    """GET /profile/users 回應。"""
    users: list[UserListItem]
    count: int


class LoginRequest(BaseModel):
    """POST /profile/login 請求：person_id + password。"""
    person_id: str
    password: str


class UserCourseItem(BaseModel):
    """User_Course_Relation 單筆課程／選課資訊（含 user_type；course_name 自 Course 表串接）。"""
    course_user_id: int
    course_id: int
    college_id: Optional[int] = None
    course_name: Optional[str] = None
    user_type: Optional[int] = None


UserListItem.model_rebuild()


class LoginResponse(BaseModel):
    """登入成功回傳使用者資訊（不含 password）；user.courses 與頂層 courses 皆為該帳號選課列。"""
    user: UserListItem
    courses: list[UserCourseItem] = Field(default_factory=list)


class UpdateProfileRequest(BaseModel):
    """PATCH /profile 請求：身分以 query person_id 為準；可選 body.person_id 須與之一致；可更新 name、user_type。"""
    person_id: Optional[str] = None
    name: Optional[str] = None
    user_type: Optional[int] = None


class UploadUserRequest(BaseModel):
    """POST /profile/users 請求：新增使用者並建立一筆選課；query person_id 須與 body.person_id 一致。"""
    person_id: str
    name: str
    course_id: int
    user_type: int


BATCH_UPLOAD_USER_TYPE = 3
BATCH_UPLOAD_DEFAULT_PASSWORD = "0000"


class BatchUserRow(BaseModel):
    """批次新增單筆：僅 person_id、name；密碼固定寫入 BATCH_UPLOAD_DEFAULT_PASSWORD（0000）。"""
    person_id: str
    name: str


class BatchUserFailure(BaseModel):
    person_id: str
    detail: str


class BatchCreateUsersResponse(BaseModel):
    """POST /profile/users/batch 回應。"""
    created: list[UserListItem]
    failed: list[BatchUserFailure]
    created_count: int
    failed_count: int


class DeleteUserRequest(BaseModel):
    """PUT /profile/users/delete：要軟刪除的使用者 person_id。"""
    person_id: str


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


def _insert_user_upload(
    supabase,
    person_id: str,
    name: str,
    user_type: int,
    *,
    college_id: int = 0,
    course_id: int = 0,
    password: str = "",
) -> UserListItem:
    exist = (
        supabase.table(USER_TABLE)
        .select("user_id")
        .eq("person_id", person_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if exist.data:
        raise HTTPException(status_code=409, detail="person_id 已存在")
    ts = now_taipei_iso()
    row_in = {
        "person_id": person_id,
        "name": (name or "").strip() or None,
        "password": password,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }
    if college_id:
        row_in["college_id"] = college_id
    ins = supabase.table(USER_TABLE).insert(row_in).execute()
    user_row = ins.data[0] if ins.data else None
    if not user_row:
        resp = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("person_id", person_id)
            .limit(1)
            .execute()
        )
        user_row = resp.data[0] if resp.data else None
    if not user_row:
        raise HTTPException(status_code=500, detail="新增使用者成功但未回傳資料")
    uid = user_row["user_id"]
    _insert_user_course_relation(
        supabase,
        user_id=uid,
        person_id=person_id,
        name=name,
        user_type=user_type,
        ts=ts,
        college_id=college_id,
        course_id=course_id,
    )
    college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(user_row.get("college_id"))])
    courses_map = _courses_for_users(supabase, [uid])
    return _user_list_item(user_row, college_map, courses_map.get(uid, []))


@router.get("/users", response_model=ListUsersResponse)
def list_users(_person_id: PersonId):
    """
    列出 User 表內容（含 password）；僅 deleted = false。新增請用 POST /profile/users 或 POST /profile/users/batch。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(f"{USER_TABLE_COLUMNS}, password")
            .eq("deleted", False)
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
                    include_password=True,
                )
                for r in rows
            ],
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


def _soft_delete_user(supabase, target_person_id: str) -> LoginResponse:
    resp = (
        supabase.table(USER_TABLE)
        .select(USER_TABLE_COLUMNS)
        .eq("person_id", target_person_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="找不到該使用者或已刪除")
    row = resp.data[0]
    user_id = row.get("user_id")
    pid = row.get("person_id")
    ts = now_taipei_iso()
    supabase.table(USER_TABLE).update({"deleted": True, "updated_at": ts}).eq("user_id", user_id).eq("person_id", pid).execute()
    supabase.table(USER_COURSE_RELATION_TABLE).update({"deleted": True, "updated_at": ts}).eq("user_id", user_id).execute()
    row_out = {**row, "deleted": True}
    college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(row_out.get("college_id"))])
    courses = _courses_for_users(supabase, [int(user_id)]).get(int(user_id), []) if user_id is not None else []
    return LoginResponse(user=_user_list_item(row_out, college_map, courses), courses=courses)


@router.put("/users/delete", response_model=LoginResponse, summary="Soft delete user", operation_id="profile_users_delete")
def soft_delete_user(
    body: openapi_body(DeleteUserRequest, {"person_id": "string"}),
    _person_id: PersonId,
):
    """
    PUT /profile/users/delete。軟刪除：將指定 person_id 之使用者 deleted 設為 true（需帶 query person_id）。
    """
    target = (body.person_id or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="person_id 不可為空")

    try:
        supabase = get_supabase()
        return _soft_delete_user(supabase, target)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users", response_model=LoginResponse)
def upload_user(
    body: openapi_body(
        UploadUserRequest,
        {
            "person_id": "string",
            "name": "string",
            "course_id": 1,
            "user_type": 3,
        },
    ),
    person_id: PersonId,
):
    """
    新增單一使用者：body 傳入 person_id、name、course_id、user_type（選課身份）；
    college_id 依 course_id 自 Course 表帶入。
    query 的 person_id 須與 body.person_id 一致。
    """
    body_pid = (body.person_id or "").strip()
    if body_pid != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    if not body.course_id:
        raise HTTPException(status_code=400, detail="course_id 不可為空")

    try:
        supabase = get_supabase()
        course_by_id = _fetch_courses_by_ids(supabase, [body.course_id])
        course_row = course_by_id.get(body.course_id)
        if not course_row:
            raise HTTPException(status_code=400, detail="找不到指定課程")
        college_id = int(course_row.get("college_id") or 0)
        if not college_id:
            raise HTTPException(status_code=400, detail="課程未設定所屬學院")
        user = _insert_user_upload(
            supabase,
            person_id,
            body.name,
            body.user_type,
            college_id=college_id,
            course_id=body.course_id,
        )
        return LoginResponse(user=user, courses=user.courses)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _batch_upload_users(supabase, rows: list[BatchUserRow]) -> BatchCreateUsersResponse:
    created: list[UserListItem] = []
    failed: list[BatchUserFailure] = []
    for row in rows:
        pid = (row.person_id or "").strip()
        if not pid:
            failed.append(
                BatchUserFailure(
                    person_id=row.person_id if row.person_id is not None else "",
                    detail="person_id 不可為空",
                )
            )
            continue
        try:
            u = _insert_user_upload(
                supabase,
                pid,
                row.name,
                BATCH_UPLOAD_USER_TYPE,
                password=BATCH_UPLOAD_DEFAULT_PASSWORD,
            )
            created.append(u)
        except HTTPException as he:
            failed.append(BatchUserFailure(person_id=pid, detail=str(he.detail)))
        except Exception as e:
            failed.append(BatchUserFailure(person_id=pid, detail=str(e)))
    return BatchCreateUsersResponse(
        created=created,
        failed=failed,
        created_count=len(created),
        failed_count=len(failed),
    )


@router.post("/users/batch", response_model=BatchCreateUsersResponse)
def batch_upload_users(
    body: Annotated[
        list[BatchUserRow],
        Body(openapi_examples={"default": {"summary": "Default", "value": [{"person_id": "string", "name": "string"}]}}),
    ],
    _person_id: PersonId,
):
    """
    批次新增使用者：body 為陣列，每筆僅 person_id、name；user_type 固定為 3；
    密碼預設為 0000（與登入 API 相同之純文字儲存）。
    已存在之 person_id 會列入 failed，其餘仍會繼續寫入。
    """
    if not body:
        raise HTTPException(status_code=400, detail="請至少傳入一筆使用者")

    try:
        supabase = get_supabase()
        return _batch_upload_users(supabase, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("", response_model=LoginResponse)
def update_profile(
    body: openapi_body(
        UpdateProfileRequest,
        {"person_id": None, "name": None, "user_type": None},
    ),
    person_id: PersonId,
):
    """
    修改個資：以 query person_id 識別使用者，可更新 name、user_type。回傳更新後的使用者資訊（不含 password）。
    """
    if (body.person_id or "").strip() and (body.person_id or "").strip() != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    if body.name is None and body.user_type is None:
        raise HTTPException(status_code=400, detail="請傳入 name 或 user_type 以進行修改")

    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("person_id", person_id)
            .or_(ACTIVE_DELETED_FILTER)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該使用者")
        row = resp.data[0]

        user_id = row.get("user_id")
        user_updates: dict = {}
        rel_updates: dict = {}
        if body.name is not None:
            user_updates["name"] = (body.name or "").strip() or None
            rel_updates["name"] = (body.name or "").strip() or ""
        if body.user_type is not None:
            rel_updates["user_type"] = body.user_type
        if not user_updates and not rel_updates:
            college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(row.get("college_id"))])
            courses = _courses_for_users(supabase, [int(user_id)]).get(int(user_id), []) if user_id is not None else []
            user = _user_list_item(row, college_map, courses)
            return LoginResponse(user=user, courses=courses)

        ts = now_taipei_iso()
        if user_updates:
            user_updates["updated_at"] = ts
            supabase.table(USER_TABLE).update(user_updates).eq("user_id", user_id).eq("person_id", person_id).execute()
        if rel_updates:
            rel_updates["updated_at"] = ts
            supabase.table(USER_COURSE_RELATION_TABLE).update(rel_updates).eq("user_id", user_id).or_(
                ACTIVE_DELETED_FILTER
            ).execute()
        resp2 = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("user_id", user_id)
            .eq("person_id", person_id)
            .or_(ACTIVE_DELETED_FILTER)
            .execute()
        )
        out_row = resp2.data[0] if resp2.data else row
        college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(out_row.get("college_id"))])
        courses = _courses_for_users(supabase, [int(user_id)]).get(int(user_id), []) if user_id is not None else []
        user = _user_list_item(out_row, college_map, courses)
        return LoginResponse(user=user, courses=courses)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=LoginResponse)
def login(
    body: openapi_body(LoginRequest, {"person_id": "string", "password": "string"}),
    person_id: PersonId,
):
    """
    以 person_id 與 password 驗證登入；成功回傳該使用者資訊（不含 password）及 User_Course_Relation 課程列表。
    query 的 person_id 須與 body.person_id 一致。
    """
    body_pid = (body.person_id or "").strip()
    if body_pid != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
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
        if not resp.data or len(resp.data) == 0:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        row = resp.data[0]
        if (row.get("password") or "").strip() != pwd:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        uid = row.get("user_id")
        courses = _courses_for_users(supabase, [int(uid)]).get(int(uid), []) if uid is not None else []
        college_map = _fetch_colleges_by_ids(supabase, [_normalize_college_id(row.get("college_id"))])
        user = _user_list_item(row, college_map, courses)
        return LoginResponse(user=user, courses=courses)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
