"""OpenAPI request body helpers：Swagger Example Value 依 dict 插入順序顯示（避開 Pydantic example 字母排序）。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body
from pydantic import BaseModel


def openapi_examples(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {"default": {"summary": "Default", "value": value}}


def openapi_body(model: type[BaseModel], example: dict[str, Any]):
    """回傳 Annotated[model, Body(openapi_examples=…)]，供路由參數型別標註。"""
    return Annotated[model, Body(openapi_examples=openapi_examples(example))]
