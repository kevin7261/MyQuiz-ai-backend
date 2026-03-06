"""使用者相關 API：列出 User 表等。"""

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.supabase_client import get_supabase

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/debug")
def debug_env():
    """暫用：確認後端讀到的 Supabase 設定（不回傳完整 key）。"""
    def mask(v: str) -> str:
        v = v.strip()
        if not v:
            return "（未設定）"
        return v[:12] + "..." + v[-4:]

    url   = (os.environ.get("SUPABASE_URL") or "").strip()
    anon  = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()
    srv   = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    sec   = (os.environ.get("SUPABASE_SECRET_KEY") or "").strip()

    active_key = srv or sec or anon  # 與 supabase_client.py 相同邏輯
    # Publishable key 填在 service_role 仍被 Supabase 當成低權限
    is_publishable = (srv or "").strip().startswith("sb_publishable")
    key_type = (
        "service_role / Secret key (HIGH PRIV)" if ((srv or sec) and not is_publishable)
        else "anon / Publishable (LOW PRIV — RLS 擋住)"
    )
    if is_publishable:
        key_type = "⚠️ 目前是 Publishable key，權限等同 anon，RLS 會擋。請改用真正的 service_role / Secret key，或在 Supabase 為 User 表加 RLS 政策允許 anon SELECT。"

    return {
        "url": url or "（未設定）",
        "anon_key":          mask(anon),
        "service_role_key":  mask(srv),
        "secret_key":        mask(sec),
        "active_key_type":   key_type,
        "active_key_prefix": (active_key[:20] + "...") if active_key else "（無）",
    }


# 與 DB 表一致（User 表）：user_id, person_id, name, user_type, llm_api_key, user_metadata, updated_at, created_at；不含 password
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


@router.get("/users", response_model=ListUsersResponse)
def list_users():
    """
    列出 User 表全部內容（不含 password 欄位）。
    """
    try:
        supabase = get_supabase()
        # 只選取對外可曝光的欄位，不回傳 password
        resp = (
            supabase.table("User")
            .select(USER_PUBLIC_COLUMNS)
            .execute()
        )
        return ListUsersResponse(users=resp.data, count=len(resp.data))
    except Exception as e:
        err = str(e).lower()
        # DNS / 連線錯誤（httpx.ConnectError 包裝 OSError，不一定是 OSError 子類別）
        if "nodename" in err or "errno 8" in err or "name or service not known" in err:
            raise HTTPException(
                status_code=503,
                detail="無法連線至 Supabase，請確認 .env 的 SUPABASE_URL 正確且網路可連線。",
            )
        # 若表名在 DB 為小寫 "user"，Supabase 可能用 "user"
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
