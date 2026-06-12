"""
個人分析 API 模組（資料存於 Person_Analysis；見 services.analysis_setting）。

對齊測驗頁「一列一 page、按 id 操作」模式：
- 一列＝一筆分析紀錄；POST 新增、PATCH 改名、DELETE 刪除、POST /{id}/llm-analysis 寫入報告，一律按主鍵。
- 分析規則存 Course_Setting（GET/PUT /rag/person-analysis-user-prompt-text）；
  結果列的 analysis_prompt_text 僅為產生報告當下的規則快照。
- person_id 一律為呼叫 API 的登入帳號。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from services.exam_queries import (
    exams_with_quizzes_response,
    quizzes_by_person_id,
)
from services.analysis_setting import (
    add_person_analysis_row,
    fetch_person_analyses_by_person,
    fetch_person_analysis_row,
    fetch_person_analysis_user_prompt_for_llm,
    resolve_login_person_id,
    save_person_analysis_result,
    soft_delete_person_analysis,
    update_person_analysis_name,
)
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.llm_key import get_person_analysis_api_key, get_rag_llm_model
from utils.openapi import openapi_body
from utils.serialization import to_json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/person-analyses", tags=["person analysis"])

ANALYSIS_LABEL_PERSON = "個人分析"


def _caller_person_id_or_404(person_id: str) -> str:
    login = resolve_login_person_id(person_id)
    if not login:
        raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
    return login


class PersonAnalysisListItem(BaseModel):
    """單筆 Person_Analysis 列（GET /person-analyses）。"""
    person_analysis_id: Optional[int] = Field(
        default=None, description="Person_Analysis 主鍵"
    )
    person_id: Optional[str] = Field(default=None, description="該列登入帳號")
    course_id: Optional[int] = None
    analysis_name: Optional[str] = Field(
        default=None, description="分析名稱（DB 欄位 analysis_name）"
    )
    analysis_user_prompt_text: Optional[str] = Field(
        default=None,
        description="教師分析指令（對應 answer_user_prompt_text）",
    )
    analysis_prompt_text: Optional[str] = Field(
        default=None,
        description="與 analysis_user_prompt_text 同源（DB 欄位 analysis_prompt_text）",
    )
    analysis_text: Optional[str] = Field(
        default=None, description="弱點報告 Markdown（對應 answer_critique）",
    )
    exams: Optional[list[dict]] = Field(
        default=None,
        description="該列課程目前已作答題目（即時自資料庫彙整），與 POST /{id}/llm-analysis 回傳 exams 同格式；無已作答題目時為空陣列",
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PersonAnalysesResponse(BaseModel):
    """GET /person-analyses 回應。"""
    person_id: str = Field(..., description="呼叫者登入帳號")
    analyses: list[PersonAnalysisListItem] = Field(
        ..., description="該使用者所有課程的 Person_Analysis 列（person_analysis_id 升冪）"
    )
    count: int


class PersonAnalysisAddResponse(BaseModel):
    """POST /person-analyses 回應。"""
    message: str
    person_analysis_id: int
    person_id: Optional[str] = Field(default=None, description="該列登入帳號")
    course_id: Optional[int] = None
    analysis_name: Optional[str] = Field(
        default=None, description="分析名稱（DB 欄位 analysis_name；未填為空字串）"
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UpdatePersonAnalysisNameRequest(BaseModel):
    """PATCH /person-analyses/{person_analysis_id}：更新 Person_Analysis 的 analysis_name。"""
    analysis_name: str = Field(..., description="新的 analysis_name；傳空字串可清除名稱")


class PersonAnalysisNameResponse(BaseModel):
    """PATCH /person-analyses/{person_analysis_id} 回應。"""
    message: str
    person_analysis_id: int
    person_id: Optional[str] = Field(default=None, description="該列登入帳號")
    course_id: Optional[int] = None
    analysis_name: Optional[str] = Field(
        default=None, description="更新後的分析名稱（DB 欄位 analysis_name）"
    )
    updated_at: Optional[str] = None


class PersonAnalysisDeleteResponse(BaseModel):
    """DELETE /person-analyses/{person_analysis_id} 回應。"""
    message: str
    person_analysis_id: int
    person_id: Optional[str] = Field(default=None, description="該列登入帳號")
    course_id: Optional[int] = None
    updated_at: Optional[str] = None


class PersonLlmAnalysisResponse(BaseModel):
    """POST /person-analyses/{person_analysis_id}/llm-analysis 回應。"""
    person_analysis_id: int = Field(..., description="報告寫入的 Person_Analysis 主鍵")
    exams: list[dict]
    count: int
    weakness_report: Optional[str] = Field(
        default=None,
        description="弱點報告：LLM message.content 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )
    llm_error: Optional[str] = Field(
        default=None,
        description="LLM 呼叫失敗或未設定 API Key 時的錯誤原因；成功時為 null",
    )
    analysis_llm_model: str = Field(
        ...,
        description="本次弱點分析實際使用的 LLM 模型（Course_Setting key=llm-model）。API Key 為 exam-api-key",
    )


@router.get("", response_model=PersonAnalysesResponse)
def list_person_analyses(person_id: PersonId):
    """
    取值：不呼叫 LLM。必填 query `person_id`（呼叫者）。
    回傳該使用者所有課程的 Person_Analysis 結果列（analysis_text 非 null）；
    每列附上該課程目前已作答題目 `exams`（抓法同 POST /{id}/llm-analysis）。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        rows = to_json_safe(fetch_person_analyses_by_person(caller))
        exams_by_course: dict[int, list[dict]] = {}
        for row in rows:
            cid = row.get("course_id")
            if cid is None or cid in exams_by_course:
                continue
            quizzes = quizzes_by_person_id(caller, course_id=cid)
            answered = [q for q in quizzes if quiz_has_answer(q)]
            exams_by_course[cid] = to_json_safe(exams_with_quizzes_response(answered))
        items = [
            PersonAnalysisListItem(
                person_analysis_id=row.get("person_analysis_id"),
                person_id=row.get("person_id"),
                course_id=row.get("course_id"),
                analysis_name=row.get("analysis_name"),
                analysis_user_prompt_text=row.get("analysis_prompt_text"),
                analysis_prompt_text=row.get("analysis_prompt_text"),
                analysis_text=row.get("analysis_text"),
                exams=exams_by_course.get(row.get("course_id")),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
            for row in rows
        ]
        return PersonAnalysesResponse(person_id=caller, analyses=items, count=len(items))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=PersonAnalysisAddResponse, status_code=201)
def add_person_analysis(
    person_id: PersonId,
    course_id: CourseId,
    analysis_name: Optional[str] = Query(
        default=None, description="分析名稱（DB 欄位 analysis_name；未填存空字串）"
    ),
):
    """
    新增一筆空白 Person_Analysis 結果列（analysis_text=''）。必填 query `person_id`（呼叫者）、`course_id`；可選 `analysis_name`。
    新增後 GET /person-analyses 會多一列；以回傳的 `person_analysis_id` 呼叫 POST /{id}/llm-analysis 將報告寫入此列。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        row = add_person_analysis_row(caller, course_id, analysis_name)
        if not row:
            raise HTTPException(
                status_code=500,
                detail=f"新增 Person_Analysis 失敗 (person_id={caller}, course_id={course_id})",
            )
        safe = to_json_safe(row)
        return PersonAnalysisAddResponse(
            message="已新增 Person_Analysis 列",
            person_analysis_id=safe.get("person_analysis_id"),
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            analysis_name=safe.get("analysis_name"),
            created_at=safe.get("created_at"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/{person_analysis_id}", response_model=PersonAnalysisNameResponse)
def update_person_analysis_name_endpoint(
    body: openapi_body(
        UpdatePersonAnalysisNameRequest,
        {"analysis_name": "新名稱"},
    ),
    person_id: PersonId,
    person_analysis_id: int = PathParam(
        ..., gt=0, description="要更新的 Person_Analysis 主鍵"
    ),
):
    """
    更新 Person_Analysis 該列 analysis_name。部分更新（目前支援 analysis_name）；僅更新 deleted=false 的列。
    """
    try:
        _caller_person_id_or_404(person_id)
        row = update_person_analysis_name(person_analysis_id, body.analysis_name)
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 person_analysis_id={person_analysis_id} 的 Person_Analysis 資料，或已刪除",
            )
        safe = to_json_safe(row)
        return PersonAnalysisNameResponse(
            message="已更新 Person_Analysis 分析名稱",
            person_analysis_id=person_analysis_id,
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            analysis_name=safe.get("analysis_name"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/{person_analysis_id}",
    response_model=PersonAnalysisDeleteResponse,
)
def delete_person_analysis(
    person_id: PersonId,
    person_analysis_id: int = PathParam(
        ..., gt=0, description="要軟刪除的 Person_Analysis 主鍵"
    ),
):
    """
    軟刪除：將 Person_Analysis 該列 deleted 設為 true。必填 query `person_id`（呼叫者）。
    刪除後 GET /person-analyses 不再回傳該列。
    """
    try:
        _caller_person_id_or_404(person_id)
        row = soft_delete_person_analysis(person_analysis_id)
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 person_analysis_id={person_analysis_id} 的 Person_Analysis 資料，或已刪除",
            )
        safe = to_json_safe(row)
        return PersonAnalysisDeleteResponse(
            message="已將 Person_Analysis 標記為刪除",
            person_analysis_id=person_analysis_id,
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{person_analysis_id}/llm-analysis", response_model=PersonLlmAnalysisResponse)
def person_llm_analysis(
    person_id: PersonId,
    course_id: CourseId,
    person_analysis_id: int = PathParam(
        ..., gt=0, description="報告要寫入的 Person_Analysis 主鍵（POST /person-analyses 建立）"
    ),
):
    """
    必填 query `person_id`（呼叫者）、`course_id`；path 帶 `person_analysis_id`（目標列）。
    依呼叫者作答產生弱點報告並按主鍵寫入指定 Person_Analysis 列（同測驗頁 llm-generate 按 id 寫入）。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        target = fetch_person_analysis_row(person_analysis_id)
        if not target:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 person_analysis_id={person_analysis_id} 的 Person_Analysis 資料，或已刪除",
            )
        if (target.get("person_id") or "") != caller:
            raise HTTPException(status_code=403, detail="無權寫入該 Person_Analysis 列")
        if target.get("course_id") is not None and int(target["course_id"]) != int(course_id):
            raise HTTPException(
                status_code=400,
                detail=f"course_id 不符：該列屬於 course_id={target['course_id']}",
            )
        quizzes = quizzes_by_person_id(caller, course_id=course_id)
        quizzes_with_answers = [q for q in quizzes if quiz_has_answer(q)]
        data = to_json_safe(exams_with_quizzes_response(quizzes_with_answers))
        analysis_llm_model = get_rag_llm_model(course_id)
        weakness_report: Optional[str] = None
        llm_error: Optional[str] = None
        if not quizzes_with_answers:
            llm_error = "無已作答或已評級題目，無法產生弱點報告（未寫入 Person_Analysis）"
        api_key = get_person_analysis_api_key(course_id)
        if not llm_error and not api_key:
            llm_error = "未設定 API Key：PUT /v1/exam/llm-api-key（Course_Setting key=exam-api-key，依 course_id）"
        elif not llm_error:
            setting_prompt = fetch_person_analysis_user_prompt_for_llm(course_id)
            weakness_report, llm_err = generate_weakness_report_md(
                to_json_safe(quizzes_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_PERSON,
                llm_model=analysis_llm_model,
            )
            if llm_err:
                llm_error = llm_err
            if weakness_report:
                saved = save_person_analysis_result(
                    person_analysis_id,
                    weakness_report,
                    analysis_prompt_text=setting_prompt,
                )
                if not saved:
                    logger.error(
                        "person_llm_analysis: LLM ok but Person_Analysis update failed "
                        "person_analysis_id=%s",
                        person_analysis_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"弱點報告已產生但寫入 Person_Analysis 失敗 "
                            f"(person_analysis_id={person_analysis_id})"
                        ),
                    )
        return PersonLlmAnalysisResponse(
            person_analysis_id=person_analysis_id,
            exams=data,
            count=len(data),
            weakness_report=weakness_report,
            llm_error=llm_error,
            analysis_llm_model=analysis_llm_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
