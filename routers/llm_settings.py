"""Bank／Quiz 共用的 LLM 設定 routes factory。

bank 與 quiz 的 /llm-api-key、/llm-model 與三個課程預設 prompt 端點除命名
（Bank／Quiz、Course_Setting key、回應 model 名）外完全同形；由本 factory 依參數產生，
routers.bank.settings_routes／routers.quiz.settings_routes 各以自家常數呼叫一次，
避免兩份檔案逐字平行維護。

**僅內部重構，不改任何 API input/output**：路徑、operation_id、summary（取自函式名）、
docstring（OpenAPI description）、回應 model 名與欄位說明皆與原手寫版逐字一致。
"""

from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import Field, create_model

from dependencies.person_id import CurrentUser, PersonId
from dependencies.course_id import CourseId

from utils.openapi import openapi_body
from utils.supabase import get_supabase
from utils.course_setting import fetch_course_setting_text
from routers.course_settings import (
    _require_active_person,
    _require_developer_or_manager_for_analysis_prompt_write,
    _upsert_setting_and_get_row,
)

# 三個課程預設 prompt 欄位（bank／quiz 共用同一組欄位名與 Swagger 範例文字）
_PROMPT_TEXT_EXAMPLES = {
    "question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。",
    "question_user_prompt_text": "請就課程內容出一道問答題。",
    "answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。",
}


def _register(router: APIRouter, method: str, path: str, fn, *, name: str, doc: str, response_model=None, operation_id: str):
    """以指定函式名／docstring 註冊路由（summary 取自函式名、description 取自 docstring，與手寫版一致）。"""
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = doc
    router.add_api_route(path, fn, methods=[method], response_model=response_model, operation_id=operation_id)


def _add_prompt_text_routes(
    router: APIRouter,
    *,
    prefix: str,
    title: str,
    group_table: str,
    field: str,
    setting_key: str,
) -> None:
    """產生某一 prompt 欄位的 GET／PUT 課程預設端點（如 /bank/question-system-prompt-text）。"""
    _add_course_setting_prompt_routes(
        router,
        prefix=prefix,
        title=title,
        path="/" + field.replace("_", "-"),
        field=field,
        setting_key=setting_key,
        get_doc=f"讀取 {group_table}.{field} 課程預設（Course_Setting key={setting_key}）。",
        put_doc=f"寫入 {group_table}.{field} 課程預設（傳空字串可清除）。",
        example=_PROMPT_TEXT_EXAMPLES.get(field, "string"),
        request_field_description=f"{group_table}.{field} 課程預設",
    )


def _add_course_setting_prompt_routes(
    router: APIRouter,
    *,
    prefix: str,
    title: str,
    path: str,
    field: str,
    setting_key: str,
    get_doc: str,
    put_doc: str,
    example: str,
    request_field_description: str,
) -> None:
    """產生某一 Course_Setting 文字欄位的 GET／PUT 端點（路徑與回應欄位名可自訂）。"""
    camel = "".join(part.capitalize() for part in field.split("_"))
    resp_model = create_model(
        f"{title}{camel}Response",
        course_id=(int, ...),
        **{field: (Optional[str], None)},
    )
    req_model = create_model(
        f"Put{title}{camel}Request",
        **{field: (str, Field(..., description=request_field_description))},
    )
    op_suffix = field

    def get_setting(caller: CurrentUser, course_id: CourseId):
        _require_active_person(caller.person_id, caller.college_id)
        text = fetch_course_setting_text(setting_key, course_id)
        return resp_model(**{"course_id": course_id, field: text or None})

    _register(
        router, "GET", path, get_setting,
        name=f"get_{prefix}_{field}_setting",
        doc=get_doc,
        response_model=resp_model,
        operation_id=f"{prefix}_get_{op_suffix}",
    )

    def put_setting(
        body: openapi_body(req_model, {field: example}),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        try:
            supabase = get_supabase()
            _upsert_setting_and_get_row(supabase, setting_key, (getattr(body, field) or "").strip(), course_id)
            text = fetch_course_setting_text(setting_key, course_id)
            return resp_model(**{"course_id": course_id, field: text or None})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    _register(
        router, "PUT", path, put_setting,
        name=f"put_{prefix}_{field}_setting",
        doc=put_doc,
        response_model=resp_model,
        operation_id=f"{prefix}_put_{op_suffix}",
    )


def build_llm_settings_router(
    *,
    prefix: str,
    title: str,
    group_table: str,
    api_key_setting_key: str,
    llm_model_setting_key: str,
    question_system_prompt_key: str,
    question_user_prompt_key: str,
    answer_user_prompt_key: str,
    api_key_exists: Callable[[int], bool],
    fetch_api_key_setting_row: Callable[[int], dict | None],
    fetch_llm_model_setting_row: Callable[[int], dict | None],
) -> APIRouter:
    """依模組常數產生 LLM 設定 router（prefix="bank"／"quiz"；title="Bank"／"Quiz"）。

    端點順序與原手寫檔一致：api-key exists → GET/PUT api-key → GET llm-model
    → 三個 prompt 課程預設 GET/PUT → PUT llm-model。
    """
    router = APIRouter(prefix=f"/{prefix}", tags=[prefix])

    ApiKeyExistsResponse = create_model(
        f"{title}ApiKeyExistsResponse",
        course_id=(int, ...),
        exists=(bool, ...),
    )
    ApiKeyResponse = create_model(
        f"{title}ApiKeyResponse",
        course_setting_id=(Optional[int], None),
        course_id=(int, ...),
        api_key=(Optional[str], None),
    )
    LlmModelResponse = create_model(
        f"{title}LlmModelResponse",
        course_setting_id=(Optional[int], None),
        course_id=(int, ...),
        llm_model=(Optional[str], None),
    )
    PutApiKeyRequest = create_model(
        f"Put{title}ApiKeyRequest",
        api_key=(str, Field("", description=f"{title} LLM API Key（寫入 Course_Setting key={api_key_setting_key}）")),
    )
    PutLlmModelRequest = create_model(
        f"Put{title}LlmModelRequest",
        llm_model=(str, Field("", description=f"{title} 出題／批改 LLM 模型（寫入 Course_Setting key={llm_model_setting_key}）")),
    )

    def get_api_key_exists(caller: CurrentUser, course_id: CourseId):
        _require_active_person(caller.person_id, caller.college_id)
        return ApiKeyExistsResponse(course_id=course_id, exists=api_key_exists(course_id))

    _register(
        router, "GET", "/llm-api-key/exists", get_api_key_exists,
        name=f"get_{prefix}_api_key_exists",
        doc=f"查詢 {title} LLM API Key 是否已設定（Course_Setting key={api_key_setting_key}）；不回傳 key 內容。",
        response_model=ApiKeyExistsResponse,
        operation_id=f"{prefix}_llm_api_key_exists",
    )

    def get_api_key_setting(person_id: PersonId, course_id: CourseId):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        row = fetch_api_key_setting_row(course_id)
        if not row:
            return ApiKeyResponse(course_id=course_id)
        value = (row.get("value") or "").strip()
        return ApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=value or None)

    _register(
        router, "GET", "/llm-api-key", get_api_key_setting,
        name=f"get_{prefix}_api_key_setting",
        doc=f"讀取 {title} LLM API Key（Course_Setting key={api_key_setting_key}）。",
        response_model=ApiKeyResponse,
        operation_id=f"{prefix}_get_llm_api_key",
    )

    def put_api_key_setting(
        body: openapi_body(PutApiKeyRequest, {"api_key": "sk-..."}),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.api_key or "").strip()
        try:
            supabase = get_supabase()
            row = _upsert_setting_and_get_row(supabase, api_key_setting_key, value_to_save, course_id)
            if not row:
                return ApiKeyResponse(course_id=course_id, api_key=value_to_save or None)
            saved = (row.get("value") or "").strip()
            return ApiKeyResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, api_key=saved or None)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    _register(
        router, "PUT", "/llm-api-key", put_api_key_setting,
        name=f"put_{prefix}_api_key_setting",
        doc=f"寫入 {title} LLM API Key（Course_Setting key={api_key_setting_key}）。",
        response_model=ApiKeyResponse,
        operation_id=f"{prefix}_put_llm_api_key",
    )

    def get_llm_model_setting(person_id: PersonId, course_id: CourseId):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        row = fetch_llm_model_setting_row(course_id)
        if not row:
            return LlmModelResponse(course_id=course_id)
        value = (row.get("value") or "").strip()
        return LlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=value or None)

    _register(
        router, "GET", "/llm-model", get_llm_model_setting,
        name=f"get_{prefix}_llm_model_setting",
        doc=f"讀取 {title} 出題／批改 LLM 模型（Course_Setting key={llm_model_setting_key}）。",
        response_model=LlmModelResponse,
        operation_id=f"{prefix}_get_llm_model",
    )

    for field, setting_key in (
        ("question_system_prompt_text", question_system_prompt_key),
        ("question_user_prompt_text", question_user_prompt_key),
        ("answer_user_prompt_text", answer_user_prompt_key),
    ):
        _add_prompt_text_routes(
            router,
            prefix=prefix,
            title=title,
            group_table=group_table,
            field=field,
            setting_key=setting_key,
        )

    def put_llm_model_setting(
        body: openapi_body(PutLlmModelRequest, {"llm_model": "gpt-5.4"}),
        person_id: PersonId,
        course_id: CourseId,
    ):
        _require_developer_or_manager_for_analysis_prompt_write(person_id, course_id)
        value_to_save = (body.llm_model or "").strip()
        try:
            supabase = get_supabase()
            row = _upsert_setting_and_get_row(supabase, llm_model_setting_key, value_to_save, course_id)
            if not row:
                return LlmModelResponse(course_id=course_id, llm_model=value_to_save or None)
            saved = (row.get("value") or "").strip()
            return LlmModelResponse(course_setting_id=row.get("course_setting_id"), course_id=course_id, llm_model=saved or None)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    _register(
        router, "PUT", "/llm-model", put_llm_model_setting,
        name=f"put_{prefix}_llm_model_setting",
        doc=f"寫入 {title} 出題／批改 LLM 模型（Course_Setting key={llm_model_setting_key}）。",
        response_model=LlmModelResponse,
        operation_id=f"{prefix}_put_llm_model",
    )

    return router
