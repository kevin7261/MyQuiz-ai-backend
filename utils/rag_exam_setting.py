"""供測驗用 RAG：System_Setting key（rag_localhost / rag_deploy）與依連線判斷本機。"""

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
