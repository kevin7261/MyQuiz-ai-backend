"""供測驗用 RAG：System_Setting key（rag_localhost / rag_deploy）與依連線判斷本機。"""

from __future__ import annotations

from fastapi import Request

RAG_EXAM_SETTING_KEY_LOCALHOST = "rag_localhost"
RAG_EXAM_SETTING_KEY_DEPLOY = "rag_deploy"


def is_localhost_request(request: Request) -> bool:
    """依連線來源是否本機。優先 X-Forwarded-For 第一跳，否則 request.client.host。"""
    host = ""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        host = xff.split(",")[0].strip()
    elif request.client:
        host = (request.client.host or "").strip()
    host = host.lower()
    if host.startswith("::ffff:"):
        host = host[7:]
    return host in ("127.0.0.1", "localhost", "::1")


def exam_rag_setting_key(request: Request) -> str:
    """本機連線對應 rag_localhost，否則 rag_deploy。"""
    return RAG_EXAM_SETTING_KEY_LOCALHOST if is_localhost_request(request) else RAG_EXAM_SETTING_KEY_DEPLOY


def fetch_exam_rag_id_from_settings(supabase, request: Request) -> tuple[str, int | None]:
    """
    依連線讀取 System_Setting 中對應 key 的 value，解析為 rag_id。
    回傳 (實際使用的 key, rag_id)；無列或無效數字則 rag_id 為 None。
    """
    key = exam_rag_setting_key(request)
    resp = (
        supabase.table("System_Setting")
        .select("value")
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if not resp.data or len(resp.data) == 0:
        return key, None
    raw = (resp.data[0].get("value") or "").strip()
    if not raw:
        return key, None
    try:
        return key, int(raw)
    except ValueError:
        return key, None


def _rag_id_from_rag_tab_id(supabase, rag_tab_id: str) -> int | None:
    tab = (rag_tab_id or "").strip()
    if not tab:
        return None
    rag_sel = (
        supabase.table("Rag")
        .select("rag_id")
        .eq("rag_tab_id", tab)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )
    if not rag_sel.data:
        return None
    try:
        rid = int(rag_sel.data[0].get("rag_id") or 0)
    except (TypeError, ValueError):
        return None
    return rid if rid > 0 else None


def resolve_exam_content_rag_id(
    supabase,
    request: Request,
    *,
    stem_rag_unit_id: int | None = None,
    rag_quiz_id: int | None = None,
) -> tuple[int | None, str | None]:
    """
    供測驗出題／批改選擇 Rag.rag_id。

    GET /exam/rag-for-exams 可列出多個 rag_tab_id 底下之單元；若僅依 System_Setting
    單一 rag_id，會與 rag_unit_id 所屬 tab 不一致。故優先：

    1. stem_rag_unit_id > 0：由 Rag_Unit.rag_tab_id → Rag.rag_id
    2. rag_quiz_id > 0：Rag_Quiz.rag_tab_id，若空則經 rag_unit_id 同上
    3. 回退 fetch_exam_rag_id_from_settings

    回傳 (rag_id, setting_key)；成功自 1／2 解析時 setting_key 為 None；
    走回退且無效時 rag_id 為 None、setting_key 為已選之設定 key（供錯誤訊息）。
    """
    ruid = 0
    if stem_rag_unit_id is not None:
        try:
            ruid = int(stem_rag_unit_id)
        except (TypeError, ValueError):
            ruid = 0
    if ruid > 0:
        ru = (
            supabase.table("Rag_Unit")
            .select("rag_tab_id")
            .eq("rag_unit_id", ruid)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if ru.data:
            found = _rag_id_from_rag_tab_id(supabase, ru.data[0].get("rag_tab_id") or "")
            if found:
                return found, None

    rqid = 0
    if rag_quiz_id is not None:
        try:
            rqid = int(rag_quiz_id)
        except (TypeError, ValueError):
            rqid = 0
    if rqid > 0:
        rq = (
            supabase.table("Rag_Quiz")
            .select("rag_tab_id, rag_unit_id")
            .eq("rag_quiz_id", rqid)
            .eq("deleted", False)
            .limit(1)
            .execute()
        )
        if rq.data:
            r0 = rq.data[0]
            tab = (r0.get("rag_tab_id") or "").strip()
            if tab:
                found = _rag_id_from_rag_tab_id(supabase, tab)
                if found:
                    return found, None
            ru2_raw = r0.get("rag_unit_id")
            try:
                ru2 = int(ru2_raw) if ru2_raw is not None else 0
            except (TypeError, ValueError):
                ru2 = 0
            if ru2 > 0:
                ru = (
                    supabase.table("Rag_Unit")
                    .select("rag_tab_id")
                    .eq("rag_unit_id", ru2)
                    .eq("deleted", False)
                    .limit(1)
                    .execute()
                )
                if ru.data:
                    found = _rag_id_from_rag_tab_id(supabase, ru.data[0].get("rag_tab_id") or "")
                    if found:
                        return found, None

    key, rid = fetch_exam_rag_id_from_settings(supabase, request)
    if rid is not None and rid > 0:
        return rid, None
    return None, key
