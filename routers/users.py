"""
使用者相關 API 模組。
提供：
- GET /user/users：列出 User 表（不含 password）
- POST /user/login：以 person_id + password 登入
- PATCH /user/profile：更新個人資料（name、user_type、llm_api_key）
"""

# 引入 Any、Optional 型別
from typing import Any, Optional

# 引入 FastAPI 的 APIRouter、HTTPException
from fastapi import APIRouter, HTTPException

from dependencies.person_id import PersonId
# 引入 Pydantic 的 BaseModel
from pydantic import BaseModel

# Supabase 客戶端
from utils.supabase_client import get_supabase
# API 回傳之時間戳改為台北時間
from utils.datetime_utils import to_taipei_iso

# 建立路由，前綴 /user
router = APIRouter(prefix="/user", tags=["user"])

# 與 DB 表一致（User 表）：user_id, person_id, name, user_type, llm_api_key, user_metadata, updated_at, created_at；不含 password。
USER_PUBLIC_COLUMNS = "user_id, person_id, name, user_type, llm_api_key, user_metadata, updated_at, created_at"

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


@router.get("/users", response_model=ListUsersResponse)
def list_users(_person_id: PersonId):
    """
    列出 User 表全部內容（不含 password 欄位）。使用者管理請讀此 API。
    """
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("User")
            .select(USER_PUBLIC_COLUMNS)
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
        if "relation" in err or "does not exist" in err:
            try:
                resp = (
                    get_supabase()
                    .table("user")
                    .select(USER_PUBLIC_COLUMNS)
                    .execute()
                )
                return ListUsersResponse(
                    users=[UserListItem(**_user_public_dict(r)) for r in resp.data],
                    count=len(resp.data),
                )
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
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
            supabase.table("User")
            .select(USER_PUBLIC_COLUMNS)
            .eq("person_id", person_id)
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

        supabase.table("User").update(updates).eq("user_id", user_id).eq("person_id", person_id).execute()
        resp2 = (
            supabase.table("User")
            .select(USER_PUBLIC_COLUMNS)
            .eq("user_id", user_id)
            .eq("person_id", person_id)
            .execute()
        )
        out_row = resp2.data[0] if resp2.data else row
        return LoginResponse(user=UserListItem(**_user_public_dict(out_row)))
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "relation" in err or "does not exist" in err:
            try:
                resp = (
                    get_supabase()
                    .table("user")
                    .select(USER_PUBLIC_COLUMNS)
                    .eq("person_id", person_id)
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
                get_supabase().table("user").update(updates).eq("user_id", user_id).eq("person_id", person_id).execute()
                resp2 = (
                    get_supabase()
                    .table("user")
                    .select(USER_PUBLIC_COLUMNS)
                    .eq("user_id", user_id)
                    .eq("person_id", person_id)
                    .execute()
                )
                out_row = resp2.data[0] if resp2.data else row
                return LoginResponse(user=UserListItem(**_user_public_dict(out_row)))
            except HTTPException:
                raise
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
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
            supabase.table("User")
            .select(cols)
            .eq("person_id", person_id)
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
        err = str(e).lower()
        if "relation" in err or "does not exist" in err:
            try:
                resp = (
                    get_supabase()
                    .table("user")
                    .select(cols)
                    .eq("person_id", person_id)
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
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
        raise HTTPException(status_code=500, detail=str(e))
