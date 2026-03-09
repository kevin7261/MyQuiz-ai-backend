"""
使用者相關 API 模組。
提供：
- GET /user/users：列出 User 表（不含 password）
- POST /user/login：以 person_id + password 登入
- PATCH /user/profile：更新個人資料（name、user_type、llm_api_key）
"""

# 引入 Any、Optional 型別
from typing import Any, Optional

# 引入 FastAPI 的 APIRouter、Header、HTTPException
from fastapi import APIRouter, Header, HTTPException
# 引入 Pydantic 的 BaseModel
from pydantic import BaseModel

# Supabase 客戶端
from utils.supabase_client import get_supabase

# 建立路由，前綴 /user
router = APIRouter(prefix="/user", tags=["user"])

# 與 DB 表一致（User 表）：user_id, person_id, name, user_type, llm_api_key, user_metadata, updated_at, created_at；不含 password。
USER_PUBLIC_COLUMNS = "user_id, person_id, name, user_type, llm_api_key, user_metadata, updated_at, created_at"


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
    """PATCH /user/profile 請求：person_id 可放 body 或 Header X-Person-Id；可更新 name、user_type、llm_api_key。"""
    person_id: Optional[str] = None
    name: Optional[str] = None
    user_type: Optional[int] = None
    llm_api_key: Optional[str] = None


@router.get("/users", response_model=ListUsersResponse)
def list_users():
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
        return ListUsersResponse(users=resp.data, count=len(resp.data))
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
                return ListUsersResponse(users=resp.data, count=len(resp.data))
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/profile", response_model=LoginResponse)
def update_profile(
    body: UpdateProfileRequest,
    x_person_id: Optional[str] = Header(None, alias="X-Person-Id", description="person_id，可取代 body 中的 person_id"),
):
    """
    修改個資：以 person_id 識別使用者（body 或 Header X-Person-Id），可更新 name、user_type、llm_api_key。
    LLM API Key 可於 body 傳入 llm_api_key 更新（空字串表示清除）。回傳更新後的使用者資訊（不含 password）。
    """
    person_id = (body.person_id or x_person_id or "").strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="請傳入 person_id（body 或 Header X-Person-Id）")
    if body.name is None and body.user_type is None and body.llm_api_key is None:
        raise HTTPException(status_code=400, detail="請傳入 name、user_type 或 llm_api_key 以進行修改")

    out_keys = ("user_id", "person_id", "name", "user_type", "llm_api_key", "user_metadata", "updated_at", "created_at")
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
            out = {k: row.get(k) for k in out_keys}
            return LoginResponse(user=out)

        supabase.table("User").update(updates).eq("user_id", user_id).eq("person_id", person_id).execute()
        resp2 = (
            supabase.table("User")
            .select(USER_PUBLIC_COLUMNS)
            .eq("user_id", user_id)
            .eq("person_id", person_id)
            .execute()
        )
        out_row = resp2.data[0] if resp2.data else row
        out = {k: out_row.get(k) for k in out_keys}
        return LoginResponse(user=out)
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
                    out = {k: row.get(k) for k in out_keys}
                    return LoginResponse(user=out)
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
                out = {k: out_row.get(k) for k in out_keys}
                return LoginResponse(user=out)
            except HTTPException:
                raise
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """
    以 person_id 與 password 驗證登入；成功回傳該使用者資訊（不含 password）。
    """
    person_id = (body.person_id or "").strip()
    pwd = (body.password or "").strip()
    cols = f"{USER_PUBLIC_COLUMNS}, password"
    out_keys = ("user_id", "person_id", "name", "user_type", "llm_api_key", "user_metadata", "updated_at", "created_at")
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
        out = {k: row.get(k) for k in out_keys}
        return LoginResponse(user=out)
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
                out = {k: row.get(k) for k in out_keys}
                return LoginResponse(user=out)
            except HTTPException:
                raise
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))
        raise HTTPException(status_code=500, detail=str(e))
