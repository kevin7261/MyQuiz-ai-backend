"""
使用者相關 API 模組。
提供：
- GET /user/users：列出 User 表（不含 password）
- POST /user/users：新增單一使用者（person_id、name、user_type）
- POST /user/users/batch：批次新增使用者（每筆僅 person_id、name；user_type 固定為 3；password 預設 0000）
- PUT /user/users/delete：軟刪除（body.person_id 指定對象，將 deleted 設為 true）
- POST /user/login：以 person_id + password 登入（成功時另回傳該帳號之 User_Course_Relation 課程列表）
- PATCH /user/profile：更新個人資料（name、user_type、llm_api_key）
"""

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, HTTPException

from dependencies.person_id import PersonId
from pydantic import BaseModel, Field

from utils.taipei_time import now_taipei_iso, to_taipei_iso
from utils.db_schema import (
    ACTIVE_DELETED_FILTER,
    USER_COURSE_RELATION_TABLE,
    USER_TABLE,
)
from utils.openapi import openapi_body
from utils.supabase import get_supabase

router = APIRouter(prefix="/user", tags=["user"])

# User 表實體欄位（user_type / llm_api_key 在 User_Course_Relation）
USER_TABLE_COLUMNS = "user_id, person_id, name, deleted, updated_at, created_at"


def _pick_primary_relation_rows(relations: list[dict]) -> dict[int, dict]:
    """每個 user_id 取 course_user_id 最小之一列，作為 API 上單一 user_type／llm_api_key 來源。"""
    best: dict[int, dict] = {}
    for r in relations:
        uid = r.get("user_id")
        if uid is None:
            continue
        cid = r.get("course_user_id")
        cur = best.get(uid)
        if cur is None:
            best[uid] = r
            continue
        cur_cid = cur.get("course_user_id")
        if cid is not None and (cur_cid is None or cid < cur_cid):
            best[uid] = r
    return best


def _fetch_relations_by_user_ids(supabase, user_ids: list[int]) -> dict[int, dict]:
    if not user_ids:
        return {}
    resp = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id, user_id, user_type, llm_api_key")
        .in_("user_id", user_ids)
        .or_(ACTIVE_DELETED_FILTER)
        .execute()
    )
    rows = resp.data or []
    return _pick_primary_relation_rows(rows)


def _fetch_relation_for_user_id(supabase, user_id: int) -> dict | None:
    return _fetch_relations_by_user_ids(supabase, [user_id]).get(user_id)


def _fetch_all_course_relations_for_user(supabase, user_id: int) -> list[dict]:
    """該 user_id 於 User_Course_Relation 之全部有效列（依 course_user_id 排序）。"""
    resp = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id, course_id, course_name, user_type")
        .eq("user_id", user_id)
        .or_(ACTIVE_DELETED_FILTER)
        .order("course_user_id")
        .execute()
    )
    return list(resp.data or [])


def _user_public_dict(user_row: dict, relation_row: dict | None = None) -> dict:
    """組出對外使用者 dict；user_type／llm_api_key 來自 User_Course_Relation（可為 None）。"""
    out = {k: user_row.get(k) for k in ("user_id", "person_id", "name", "updated_at", "created_at")}
    if relation_row:
        out["user_type"] = relation_row.get("user_type")
        out["llm_api_key"] = relation_row.get("llm_api_key")
    else:
        out["user_type"] = None
        out["llm_api_key"] = None
    out["user_metadata"] = None
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
    llm_api_key: str = "",
) -> None:
    supabase.table(USER_COURSE_RELATION_TABLE).insert({
        "user_id": user_id,
        "person_id": person_id,
        "name": (name or "").strip() or "",
        "course_id": 0,
        "course_name": "",
        "user_type": user_type,
        "llm_api_key": llm_api_key,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }).execute()


class UserListItem(BaseModel):
    """單筆使用者（不含 password）。"""
    user_id: int
    person_id: Optional[str] = None
    name: Optional[str] = None
    user_type: Optional[int] = None
    llm_api_key: Optional[str] = None
    user_metadata: Optional[Any] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None


class ListUsersResponse(BaseModel):
    """GET /user/users 回應。"""
    users: list[UserListItem]
    count: int


class LoginRequest(BaseModel):
    """POST /user/login 請求：person_id + password。"""
    person_id: str
    password: str


class UserCourseItem(BaseModel):
    """User_Course_Relation 單筆課程／選課資訊（不含 llm_api_key）。"""
    course_user_id: int
    course_id: int
    course_name: Optional[str] = None
    user_type: Optional[int] = None


class LoginResponse(BaseModel):
    """登入成功回傳使用者資訊（不含 password）；login 時 courses 為該帳號選課列，其餘使用同一 schema 之端點為空列表。"""
    user: UserListItem
    courses: list[UserCourseItem] = Field(default_factory=list)


class UpdateProfileRequest(BaseModel):
    """PATCH /user/profile 請求：身分以 query person_id 為準；可選 body.person_id 須與之一致；可更新 name、user_type、llm_api_key。"""
    person_id: Optional[str] = None
    name: Optional[str] = None
    user_type: Optional[int] = None
    llm_api_key: Optional[str] = None


class UploadUserRequest(BaseModel):
    """POST /user/users 請求：新增使用者；query person_id 須與 body.person_id 一致。password 由 DB 寫入空字串（可日後另行更新）。"""
    person_id: str
    name: str
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
    """POST /user/users/batch 回應。"""
    created: list[UserListItem]
    failed: list[BatchUserFailure]
    created_count: int
    failed_count: int


class DeleteUserRequest(BaseModel):
    """PUT /user/users/delete：要軟刪除的使用者 person_id。"""
    person_id: str


def _insert_user_upload(
    supabase,
    person_id: str,
    name: str,
    user_type: int,
    *,
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
        llm_api_key="",
    )
    rel = _fetch_relation_for_user_id(supabase, uid)
    return UserListItem(**_user_public_dict(user_row, rel))


@router.get("/users", response_model=ListUsersResponse)
def list_users(_person_id: PersonId):
    """
    列出 User 表內容（不含 password）；僅 deleted = false。新增請用 POST /user/users 或 POST /user/users/batch。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(USER_TABLE_COLUMNS)
            .eq("deleted", False)
            .execute()
        )
        rows = resp.data or []
        uids = [r["user_id"] for r in rows if r.get("user_id") is not None]
        rel_by_uid = _fetch_relations_by_user_ids(supabase, uids)
        return ListUsersResponse(
            users=[
                UserListItem(**_user_public_dict(r, rel_by_uid.get(r["user_id"])))
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
    rel = _fetch_relation_for_user_id(supabase, int(user_id)) if user_id is not None else None
    ts = now_taipei_iso()
    supabase.table(USER_TABLE).update({"deleted": True, "updated_at": ts}).eq("user_id", user_id).eq("person_id", pid).execute()
    supabase.table(USER_COURSE_RELATION_TABLE).update({"deleted": True, "updated_at": ts}).eq("user_id", user_id).execute()
    row_out = {**row, "deleted": True}
    return LoginResponse(user=UserListItem(**_user_public_dict(row_out, rel)))


@router.put("/users/delete", response_model=LoginResponse, summary="Soft delete user", operation_id="user_users_delete")
def soft_delete_user(
    body: openapi_body(DeleteUserRequest, {"person_id": "string"}),
    _person_id: PersonId,
):
    """
    PUT /user/users/delete。軟刪除：將指定 person_id 之使用者 deleted 設為 true（需帶 query person_id）。
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
    body: openapi_body(UploadUserRequest, {"person_id": "string", "name": "string", "user_type": 3}),
    person_id: PersonId,
):
    """
    新增單一使用者：body 傳入 person_id、name、user_type。
    query 的 person_id 須與 body.person_id 一致。
    """
    body_pid = (body.person_id or "").strip()
    if body_pid != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")

    try:
        supabase = get_supabase()
        user = _insert_user_upload(supabase, person_id, body.name, body.user_type)
        return LoginResponse(user=user)
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


@router.patch("/profile", response_model=LoginResponse)
def update_profile(
    body: openapi_body(
        UpdateProfileRequest,
        {"person_id": None, "name": None, "user_type": None, "llm_api_key": None},
    ),
    person_id: PersonId,
):
    """
    修改個資：以 query person_id 識別使用者，可更新 name、user_type、llm_api_key。
    LLM API Key 可於 body 傳入 llm_api_key 更新（空字串表示清除）。回傳更新後的使用者資訊（不含 password）。
    """
    if (body.person_id or "").strip() and (body.person_id or "").strip() != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    if body.name is None and body.user_type is None and body.llm_api_key is None:
        raise HTTPException(status_code=400, detail="請傳入 name、user_type 或 llm_api_key 以進行修改")

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
        if body.llm_api_key is not None:
            rel_updates["llm_api_key"] = (body.llm_api_key or "").strip() or ""
        if not user_updates and not rel_updates:
            rel = _fetch_relation_for_user_id(supabase, int(user_id)) if user_id is not None else None
            return LoginResponse(user=UserListItem(**_user_public_dict(row, rel)))

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
        rel = _fetch_relation_for_user_id(supabase, int(user_id)) if user_id is not None else None
        return LoginResponse(user=UserListItem(**_user_public_dict(out_row, rel)))
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
        rel = _fetch_relation_for_user_id(supabase, int(uid)) if uid is not None else None
        course_rows = _fetch_all_course_relations_for_user(supabase, int(uid)) if uid is not None else []
        courses = [
            UserCourseItem(
                course_user_id=int(r["course_user_id"]),
                course_id=int(r.get("course_id") or 0),
                course_name=r.get("course_name"),
                user_type=r.get("user_type"),
            )
            for r in course_rows
            if r.get("course_user_id") is not None
        ]
        return LoginResponse(user=UserListItem(**_user_public_dict(row, rel)), courses=courses)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
