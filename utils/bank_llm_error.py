"""Bank 專屬 LLM 呼叫錯誤格式化（自 utils.llm_error 複製，與 rag 無關）。供 API 以 HTTP 200 + llm_error 回傳前端。"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import Response


class LlmCallError(Exception):
    """OpenAI／Chat Completions 呼叫失敗；路由層可轉成 llm_error 欄位。"""


def is_llm_call_error(exc: BaseException) -> bool:
    if isinstance(exc, LlmCallError):
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True
    mod = type(exc).__module__
    return mod == "openai" or mod.startswith("openai.")


def format_llm_error(exc: BaseException) -> str:
    if isinstance(exc, LlmCallError):
        return str(exc) or "LLM 呼叫失敗"
    text = str(exc).strip()
    return text or "LLM 呼叫失敗"


def llm_error_json_response(payload: dict[str, Any]) -> Response:
    """HTTP 200 JSON；payload 須含 llm_error。"""
    return Response(
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        media_type="application/json; charset=utf-8",
    )
