"""Bank 專屬：依連線來源判斷是否本機（自 utils.rag_exam_setting.is_localhost_request 複製，與 rag 無關）。"""

from __future__ import annotations

from fastapi import Request


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
