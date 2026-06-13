"""
測驗課程分析 API 模組（分析整門課程所有學生的 Quiz_QA 作答；結果存於 Quiz_Analysis）。

定位等同 course_analysis 之於 Exam_Quiz，差異在於：
- 資料來源：Quiz_QA（bank 出題試卷），而非 Exam_Quiz（RAG 出題試卷）
- 結果表：Quiz_Analysis（非 Course_Analysis）
- API Key：quiz-analysis-api-key（非 quiz-api-key）
- LLM 模型：quiz-analysis-llm-model

對齊「一列一 page、按 id 操作」模式：
- 一列＝一筆分析紀錄；POST 新增、PATCH 改名、DELETE 刪除、POST /{id}/llm-analysis 寫入報告。
- 分析指令存 Course_Setting（GET/PUT /quiz-analyses/analysis-user-prompt-text）；
  結果列的 analysis_prompt_text 為產生報告當下的規則快照。
- person_id 一律為呼叫 API 的登入帳號（建立者／教師）。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import PersonId
from routers.analysis_prompt_settings import (
    register_analysis_llm_api_key_routes,
    register_analysis_llm_model_routes,
    register_analysis_user_prompt_routes,
)
from services.analysis_setting import resolve_login_person_id
from services.quiz_analysis_setting import (
    add_quiz_analysis_row,
    fetch_quiz_analyses_by_course,
    fetch_quiz_analysis_row,
    fetch_quiz_analysis_user_prompt_for_llm,
    save_quiz_analysis_result,
    soft_delete_quiz_analysis,
    update_quiz_analysis_name,
)
from services.quiz_queries import quiz_qas_by_course_id, quizzes_with_qas_response
from services.weakness_report import generate_weakness_report_md, quiz_has_answer
from utils.openapi import openapi_body
from utils.analysis_llm_key import (
    fetch_quiz_analysis_llm_model_setting_row,
    get_quiz_analysis_api_key,
    get_quiz_analysis_llm_model,
    quiz_analysis_api_key_exists,
)
from utils.course_setting import (
    COURSE_SETTING_QUIZ_ANALYSIS_API_KEY,
    COURSE_SETTING_QUIZ_ANALYSIS_LLM_MODEL,
    COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT,
)
from utils.serialization import to_json_safe
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz-analyses", tags=["quiz analysis"])

register_analysis_llm_api_key_routes(
    router,
    setting_key=COURSE_SETTING_QUIZ_ANALYSIS_API_KEY,
    operation_id_prefix="quiz_analysis",
    title="Quiz Analysis",
    api_key_exists=quiz_analysis_api_key_exists,
)
register_analysis_llm_model_routes(
    router,
    setting_key=COURSE_SETTING_QUIZ_ANALYSIS_LLM_MODEL,
    operation_id_prefix="quiz_analysis",
    title="Quiz Analysis",
    fetch_llm_model_setting_row=fetch_quiz_analysis_llm_model_setting_row,
)
register_analysis_user_prompt_routes(
    router,
    setting_key=COURSE_SETTING_QUIZ_ANALYSIS_USER_PROMPT_TEXT,
    operation_id_prefix="quiz_analysis",
    example="請彙整全課程學生作答弱點…",
)

ANALYSIS_LABEL_QUIZ = "測驗課程分析"


def _caller_person_id_or_404(person_id: str) -> str:
    login = resolve_login_person_id(person_id)
    if not login:
        raise HTTPException(status_code=404, detail=f"找不到使用者 person_id={person_id}")
    return login


class QuizAnalysisListItem(BaseModel):
    """單筆 Quiz_Analysis 列（GET /quiz-analyses）。"""
    quiz_analysis_id: Optional[int] = Field(
        default=None, description="Quiz_Analysis 主鍵"
    )
    person_id: Optional[str] = Field(default=None, description="建立者登入帳號")
    course_id: Optional[int] = None
    analysis_name: Optional[str] = Field(
        default=None, description="分析名稱（DB 欄位 analysis_name）"
    )
    analysis_user_prompt_text: Optional[str] = Field(
        ..., description="產生報告當下的教師指令快照"
    )
    analysis_prompt_text: Optional[str] = Field(
        ..., description="與 analysis_user_prompt_text 同源（DB 欄位 analysis_prompt_text）"
    )
    analysis_text: Optional[str] = Field(
        default=None, description="分析報告 Markdown"
    )
    quizzes: Optional[list[dict]] = Field(
        default=None,
        description=(
            "該課程目前已作答 Quiz_QA 依 quiz_page_id → quiz_group_id 分組（即時自資料庫彙整，全課程共用）；"
            "無已作答題目時為空陣列"
        ),
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class QuizAnalysesResponse(BaseModel):
    """GET /quiz-analyses 回應。"""
    course_id: int
    analyses: list[QuizAnalysisListItem] = Field(
        ..., description="該課程所有 Quiz_Analysis 列（quiz_analysis_id 升冪）"
    )
    count: int


class QuizAnalysisAddResponse(BaseModel):
    """POST /quiz-analyses 回應。"""
    message: str
    quiz_analysis_id: int
    person_id: Optional[str] = Field(default=None, description="建立者登入帳號")
    course_id: Optional[int] = None
    analysis_name: Optional[str] = Field(
        default=None, description="分析名稱（未填為空字串）"
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UpdateQuizAnalysisNameRequest(BaseModel):
    """PATCH /quiz-analyses/{quiz_analysis_id}：更新 Quiz_Analysis 的 analysis_name。"""
    analysis_name: str = Field(..., description="新的 analysis_name；傳空字串可清除名稱")


class QuizAnalysisNameResponse(BaseModel):
    """PATCH /quiz-analyses/{quiz_analysis_id} 回應。"""
    message: str
    quiz_analysis_id: int
    person_id: Optional[str] = None
    course_id: Optional[int] = None
    analysis_name: Optional[str] = None
    updated_at: Optional[str] = None


class QuizAnalysisDeleteResponse(BaseModel):
    """DELETE /quiz-analyses/{quiz_analysis_id} 回應。"""
    message: str
    quiz_analysis_id: int
    person_id: Optional[str] = None
    course_id: Optional[int] = None
    updated_at: Optional[str] = None


class QuizLlmAnalysisResponse(BaseModel):
    """POST /quiz-analyses/{quiz_analysis_id}/llm-analysis 回應。"""
    quiz_analysis_id: int = Field(..., description="報告寫入的 Quiz_Analysis 主鍵")
    quizzes: list[dict] = Field(
        ...,
        description="已作答 Quiz_QA 依 quiz_page_id → quiz_group_id 分組（全課程）",
    )
    count: int = Field(..., description="試卷數（quiz_page_id 個數）")
    weakness_report: Optional[str] = Field(
        default=None,
        description="分析報告：LLM message.content 原文 Markdown；未設定 API Key、呼叫失敗或無內容時為 null",
    )
    llm_error: Optional[str] = Field(
        default=None,
        description="LLM 呼叫失敗或未設定 API Key 時的錯誤原因；成功時為 null",
    )
    analysis_llm_model: str = Field(
        ...,
        description="本次分析實際使用的 LLM 模型（Course_Setting key=quiz-analysis-llm-model）。API Key 為 quiz-analysis-api-key",
    )


@router.get("", response_model=QuizAnalysesResponse)
def list_quiz_analyses(person_id: PersonId, course_id: CourseId):
    """
    取值：不呼叫 LLM。必填 query `person_id`（呼叫者）、`course_id`。
    回傳該課程所有 Quiz_Analysis 結果列；
    每列附上該課程目前已作答 Quiz_QA 彙整（`quizzes`，格式同 POST /{id}/llm-analysis 回傳）。
    """
    try:
        _caller_person_id_or_404(person_id)
        rows = to_json_safe(fetch_quiz_analyses_by_course(course_id))
        qas = quiz_qas_by_course_id(course_id)
        answered = [q for q in qas if quiz_has_answer(q)]
        course_quizzes = to_json_safe(quizzes_with_qas_response(answered))
        items = [
            QuizAnalysisListItem(
                quiz_analysis_id=row.get("quiz_analysis_id"),
                person_id=row.get("person_id"),
                course_id=row.get("course_id"),
                analysis_name=row.get("analysis_name"),
                analysis_user_prompt_text=row.get("analysis_prompt_text"),
                analysis_prompt_text=row.get("analysis_prompt_text"),
                analysis_text=row.get("analysis_text"),
                quizzes=course_quizzes,
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
            for row in rows
        ]
        return QuizAnalysesResponse(
            course_id=int(course_id), analyses=items, count=len(items)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=QuizAnalysisAddResponse, status_code=201)
def add_quiz_analysis(
    person_id: PersonId,
    course_id: CourseId,
    analysis_name: Optional[str] = Query(
        default=None, description="分析名稱（未填存空字串）"
    ),
):
    """
    新增一筆空白 Quiz_Analysis 結果列（analysis_text=''）。
    必填 query `person_id`（呼叫者）、`course_id`；可選 `analysis_name`。
    新增後 GET /quiz-analyses 會多一列；以回傳的 `quiz_analysis_id` 呼叫 POST /{id}/llm-analysis 將報告寫入此列。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        row = add_quiz_analysis_row(caller, course_id, analysis_name)
        if not row:
            raise HTTPException(
                status_code=500,
                detail=f"新增 Quiz_Analysis 失敗 (person_id={caller}, course_id={course_id})",
            )
        safe = to_json_safe(row)
        return QuizAnalysisAddResponse(
            message="已新增 Quiz_Analysis 列",
            quiz_analysis_id=safe.get("quiz_analysis_id"),
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


@router.patch("/{quiz_analysis_id}", response_model=QuizAnalysisNameResponse)
def update_quiz_analysis_name_endpoint(
    body: openapi_body(
        UpdateQuizAnalysisNameRequest,
        {"analysis_name": "新名稱"},
    ),
    person_id: PersonId,
    quiz_analysis_id: int = PathParam(
        ..., gt=0, description="要更新的 Quiz_Analysis 主鍵"
    ),
):
    """
    更新 Quiz_Analysis 該列 analysis_name。部分更新（目前支援 analysis_name）；僅更新 deleted=false 的列。
    """
    try:
        _caller_person_id_or_404(person_id)
        row = update_quiz_analysis_name(quiz_analysis_id, body.analysis_name)
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 quiz_analysis_id={quiz_analysis_id} 的 Quiz_Analysis 資料，或已刪除",
            )
        safe = to_json_safe(row)
        return QuizAnalysisNameResponse(
            message="已更新 Quiz_Analysis 分析名稱",
            quiz_analysis_id=quiz_analysis_id,
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            analysis_name=safe.get("analysis_name"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{quiz_analysis_id}", response_model=QuizAnalysisDeleteResponse)
def delete_quiz_analysis(
    person_id: PersonId,
    quiz_analysis_id: int = PathParam(
        ..., gt=0, description="要軟刪除的 Quiz_Analysis 主鍵"
    ),
):
    """
    軟刪除：將 Quiz_Analysis 該列 deleted 設為 true。必填 query `person_id`（呼叫者）。
    刪除後 GET /quiz-analyses 不再回傳該列。
    """
    try:
        _caller_person_id_or_404(person_id)
        row = soft_delete_quiz_analysis(quiz_analysis_id)
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 quiz_analysis_id={quiz_analysis_id} 的 Quiz_Analysis 資料，或已刪除",
            )
        safe = to_json_safe(row)
        return QuizAnalysisDeleteResponse(
            message="已將 Quiz_Analysis 標記為刪除",
            quiz_analysis_id=quiz_analysis_id,
            person_id=safe.get("person_id"),
            course_id=safe.get("course_id"),
            updated_at=safe.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{quiz_analysis_id}/llm-analysis", response_model=QuizLlmAnalysisResponse)
def quiz_llm_analysis(
    person_id: PersonId,
    course_id: CourseId,
    quiz_analysis_id: int = PathParam(
        ..., gt=0, description="報告要寫入的 Quiz_Analysis 主鍵（POST /quiz-analyses 建立）"
    ),
):
    """
    必填 query `person_id`（呼叫者）、`course_id`；path 帶 `quiz_analysis_id`（目標列）。
    分析指令自 Course_Setting key=quiz_analysis_user_prompt_text 讀取
    （GET/PUT /quiz-analyses/analysis-user-prompt-text）。
    彙整課程內所有學生的 Quiz_QA 作答紀錄，使用 LLM 產生測驗課程分析報告並寫入指定 Quiz_Analysis 列。
    API Key 使用 Course_Setting key=quiz-analysis-api-key；模型使用 key=quiz-analysis-llm-model。
    """
    try:
        caller = _caller_person_id_or_404(person_id)
        target = fetch_quiz_analysis_row(quiz_analysis_id)
        if not target:
            raise HTTPException(
                status_code=404,
                detail=f"找不到 quiz_analysis_id={quiz_analysis_id} 的 Quiz_Analysis 資料，或已刪除",
            )
        if (target.get("person_id") or "") != caller:
            raise HTTPException(status_code=403, detail="無權寫入該 Quiz_Analysis 列")
        if target.get("course_id") is not None and int(target["course_id"]) != int(course_id):
            raise HTTPException(
                status_code=400,
                detail=f"course_id 不符：該列屬於 course_id={target['course_id']}",
            )

        qas = quiz_qas_by_course_id(course_id)
        qas_with_answers = [q for q in qas if quiz_has_answer(q)]
        quizzes_data = to_json_safe(quizzes_with_qas_response(qas_with_answers))
        analysis_llm_model = get_quiz_analysis_llm_model(course_id)

        weakness_report: Optional[str] = None
        llm_error: Optional[str] = None

        if not qas_with_answers:
            llm_error = "無已作答或已評級 Quiz_QA，無法產生分析報告（未寫入 Quiz_Analysis）"

        api_key = get_quiz_analysis_api_key(course_id)
        if not llm_error and not api_key:
            llm_error = (
                "未設定 API Key：PUT /v1/quiz-analyses/llm-api-key"
                "（Course_Setting key=quiz-analysis-api-key，依 course_id）"
            )
        elif not llm_error:
            setting_prompt = fetch_quiz_analysis_user_prompt_for_llm(course_id)
            weakness_report, llm_err = generate_weakness_report_md(
                to_json_safe(qas_with_answers),
                api_key,
                setting_prompt,
                analysis_label=ANALYSIS_LABEL_QUIZ,
                llm_model=analysis_llm_model,
            )
            if llm_err:
                llm_error = llm_err
            if weakness_report:
                saved = save_quiz_analysis_result(
                    quiz_analysis_id,
                    weakness_report,
                    analysis_prompt_text=setting_prompt,
                )
                if not saved:
                    logger.error(
                        "quiz_llm_analysis: LLM ok but Quiz_Analysis update failed "
                        "quiz_analysis_id=%s",
                        quiz_analysis_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"分析報告已產生但寫入 Quiz_Analysis 失敗 "
                            f"(quiz_analysis_id={quiz_analysis_id})"
                        ),
                    )

        return QuizLlmAnalysisResponse(
            quiz_analysis_id=quiz_analysis_id,
            quizzes=quizzes_data,
            count=len(quizzes_data),
            weakness_report=weakness_report,
            llm_error=llm_error,
            analysis_llm_model=analysis_llm_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
