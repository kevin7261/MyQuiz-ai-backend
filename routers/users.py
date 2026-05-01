"""
使用者相關 API 模組。
提供：
- GET /user/users：列出 User 表（不含 password）
- POST /user/users：新增單一使用者（person_id、name、user_type）
- POST /user/users/batch：批次新增使用者（每筆僅 person_id、name；user_type 固定為 3；password 預設 0000）
- PUT /user/users/delete：軟刪除（body.person_id 指定對象，將 deleted 設為 true）
- POST /user/login：以 person_id + password 登入
- PATCH /user/profile：更新個人資料（name、user_type、llm_api_key）
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from dependencies.person_id import PersonId
from pydantic import BaseModel

from utils.datetime_utils import now_taipei_iso, to_taipei_iso
from utils.db_tables import USER_TABLE
from utils.supabase_client import get_supabase

router = APIRouter(prefix="/user", tags=["user"])

# 查詢 User 表時的公開欄位（不含 password）
USER_PUBLIC_COLUMNS = "user_id, person_id, name, user_type, llm_api_key, user_metadata, deleted, updated_at, created_at"

# deleted=false 或 null 皆視為有效帳號（相容舊列）
ACTIVE_USER_DELETED_FILTER = "deleted.eq.false,deleted.is.null"

USER_OUT_KEYS = (
    "user_id",
    "person_id",
    "name",
    "user_type",
    "llm_api_key",
    "user_metadata",
    "updated_at",
    "created_at",
)


def _user_public_dict(row: dict) -> dict:
    """組出對外使用者 dict，updated_at / created_at 為台北時間 ISO 字串。"""
    out = {k: row.get(k) for k in USER_OUT_KEYS}
    out["updated_at"] = to_taipei_iso(out.get("updated_at"))
    out["created_at"] = to_taipei_iso(out.get("created_at"))
    return out


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


class LoginResponse(BaseModel):
    """登入成功回傳使用者資訊（不含 password）。"""
    user: UserListItem


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
        .or_(ACTIVE_USER_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if exist.data:
        raise HTTPException(status_code=409, detail="person_id 已存在")
    ts = now_taipei_iso()
    row_in = {
        "person_id": person_id,
        "name": (name or "").strip() or None,
        "user_type": user_type,
        "password": password,
        "deleted": False,
        "updated_at": ts,
        "created_at": ts,
    }
    ins = supabase.table(USER_TABLE).insert(row_in).execute()
    if ins.data and len(ins.data) > 0:
        return UserListItem(**_user_public_dict(ins.data[0]))
    resp = (
        supabase.table(USER_TABLE)
        .select(USER_PUBLIC_COLUMNS)
        .eq("person_id", person_id)
        .limit(1)
        .execute()
    )
    if resp.data and len(resp.data) > 0:
        return UserListItem(**_user_public_dict(resp.data[0]))
    raise HTTPException(status_code=500, detail="新增使用者成功但未回傳資料")


@router.get("/users", response_model=ListUsersResponse)
def list_users(_person_id: PersonId):
    """
    列出 User 表內容（不含 password）；僅 deleted = false。新增請用 POST /user/users 或 POST /user/users/batch。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(USER_PUBLIC_COLUMNS)
            .eq("deleted", False)
            .execute()
        )
        return ListUsersResponse(
            users=[UserListItem(**_user_public_dict(r)) for r in resp.data],
            count=len(resp.data),
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
        .select(USER_PUBLIC_COLUMNS)
        .eq("person_id", target_person_id)
        .or_(ACTIVE_USER_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="找不到該使用者或已刪除")
    row = resp.data[0]
    user_id = row.get("user_id")
    pid = row.get("person_id")
    supabase.table(USER_TABLE).update({"deleted": True, "updated_at": now_taipei_iso()}).eq("user_id", user_id).eq("person_id", pid).execute()
    row_out = {**row, "deleted": True}
    return LoginResponse(user=UserListItem(**_user_public_dict(row_out)))


@router.put("/users/delete", response_model=LoginResponse, summary="Soft delete user", operation_id="user_users_delete")
def soft_delete_user(body: DeleteUserRequest, _person_id: PersonId):
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
def upload_user(body: UploadUserRequest, person_id: PersonId):
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
def batch_upload_users(body: list[BatchUserRow], _person_id: PersonId):
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
    body: UpdateProfileRequest,
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
            .select(USER_PUBLIC_COLUMNS)
            .eq("person_id", person_id)
            .or_(ACTIVE_USER_DELETED_FILTER)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            raise HTTPException(status_code=404, detail="找不到該使用者")
        row = resp.data[0]

        user_id = row.get("user_id")
        updates = {}
        if body.name is not None:
            updates["name"] = (body.name or "").strip() or None
        if body.user_type is not None:
            updates["user_type"] = body.user_type
        if body.llm_api_key is not None:
            updates["llm_api_key"] = (body.llm_api_key or "").strip() or None
        if not updates:
            return LoginResponse(user=UserListItem(**_user_public_dict(row)))

        updates["updated_at"] = now_taipei_iso()
        supabase.table(USER_TABLE).update(updates).eq("user_id", user_id).eq("person_id", person_id).execute()
        resp2 = (
            supabase.table(USER_TABLE)
            .select(USER_PUBLIC_COLUMNS)
            .eq("user_id", user_id)
            .eq("person_id", person_id)
            .or_(ACTIVE_USER_DELETED_FILTER)
            .execute()
        )
        out_row = resp2.data[0] if resp2.data else row
        return LoginResponse(user=UserListItem(**_user_public_dict(out_row)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, person_id: PersonId):
    """
    以 person_id 與 password 驗證登入；成功回傳該使用者資訊（不含 password）。
    query 的 person_id 須與 body.person_id 一致。
    """
    body_pid = (body.person_id or "").strip()
    if body_pid != person_id:
        raise HTTPException(status_code=400, detail="body 的 person_id 與 query 不一致")
    pwd = (body.password or "").strip()
    cols = f"{USER_PUBLIC_COLUMNS}, password"
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_TABLE)
            .select(cols)
            .eq("person_id", person_id)
            .or_(ACTIVE_USER_DELETED_FILTER)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        row = resp.data[0]
        if (row.get("password") or "").strip() != pwd:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        return LoginResponse(user=UserListItem(**_user_public_dict(row)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
