"""routers.exam schemas（自 exam.py 拆分）。"""

from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


from utils.db_schema import (
    QUIZ_HISTORY_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM,
    coerce_quiz_history_prompt_text_request,
    coerce_quiz_history_request,
)


# ---------------------------------------------------------------------------
# 模型與檢索常數
# ---------------------------------------------------------------------------
# 本模組不直接宣告 OpenAI 模型名；出題呼叫 `services.quiz_generation`（`QUIZ_LLM_MODEL`、embedding、k），
# 批改呼叫 `services.grading`（`GRADE_LLM_MODEL`＝`QUIZ_LLM_MODEL`、GET/PUT /rag/llm_model、檢索與 chunk 常數）。


# ---------------------------------------------------------------------------
# LLM Prompt（exam 出題 user 前綴；置於課程內容／檢索片段之前）
# ---------------------------------------------------------------------------
# 由 _exam_llm_generate_api_instruction 組字；欄位名與順序同 public.Exam_Quiz（至 quiz_user_prompt_text）。

ExamQuizRateValue = Literal[-1, 0, 1]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ListExamResponse(BaseModel):
    """GET /exam/pages 回應：每筆 Exam 含 quizzes[]（Exam_Quiz，含 follow_up 鏈、quiz_history_list、quiz_rate、grade_rate）。"""
    exams: list[dict] = Field(
        ...,
        description="每筆 Exam 的 quizzes[] 為 Exam_Quiz（含 follow_up、follow_up_quiz 鏈、quiz_rate、grade_rate、答案欄位）",
    )
    count: int


class ListRagForExamsResponse(BaseModel):
    """GET /exam/rag-for-exams：單元為完整 Rag_Unit；quizzes 含 RAG 關聯鍵與出題／批改 prompt。"""
    units: list[dict] = Field(
        ...,
        description="Rag_Unit 列；每筆 quizzes[] 含 follow_up、rag_quiz_id、rag_page_id、rag_unit_id、person_id、quiz_name、quiz_user_prompt_text、quiz_content、quiz_hint、quiz_answer_reference、answer_user_prompt_text",
    )
    count: int


class CreateExamRequest(BaseModel):
    """POST /exam/page/create：欄位順序同 public.Exam（exam_page_id, person_id, tab_name, local；不含 exam_id／course_id／deleted／時間戳）。"""
    exam_page_id: str | None = Field(None, description="選填；未傳則由後端產生")
    person_id: str = Field("", description="選填，寫入 Exam.person_id")
    tab_name: str = Field("", description="測驗顯示名稱")
    local: bool = Field(False, description="是否為本機 Exam")


class UpdateExamUnitNameRequest(BaseModel):
    """PUT /exam/page/tab-name：以 exam_id（主鍵）更新 tab_name。"""
    exam_id: int = Field(..., description="Exam 主鍵")
    tab_name: str = Field(..., description="新的顯示名稱")


class ExamCreateQuizRequest(BaseModel):
    """POST /exam/page/quiz/create：新增空白 Exam_Quiz（無 LLM）。僅 exam_page_id（不傳 rag_unit_id）。"""
    exam_page_id: str = Field("", description="目標 Exam 的 exam_page_id")


class ExamQuizHistoryPair(BaseModel):
    """quiz_history_list 單筆：八欄位物件。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_OPENAPI_ITEM]},
    )

    rag_unit_id: int = Field(0, ge=0, description="Rag_Unit 主鍵", examples=[1])
    quiz_name: str = Field("", description="題型名稱")
    follow_up: bool = Field(
        False,
        description="是否為追問題",
        validation_alias=AliasChoices("follow_up", "followup", "followUp"),
    )
    quiz_content: str = Field(..., description="先前題目題幹")
    quiz_hint: str = Field("", description="提示")
    answer_content: str = Field(
        "",
        description="先前作答（學生答案）",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    quiz_answer_reference: str = Field(
        "",
        description="該題參考答案（對齊 quiz_answer_reference）",
        validation_alias=AliasChoices(
            "quiz_answer_reference",
            "quiz_reference_answer",
            "reference_answer",
        ),
    )
    answer_critique: str = Field(
        "",
        description="該題評閱／批改評語（對齊 answer_critique）",
        validation_alias=AliasChoices("answer_critique", "critique", "quiz_comments"),
    )


def _coerce_exam_quiz_history_list_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list（不讀 DB）。"""
    return coerce_quiz_history_request(v)


def _coerce_exam_quiz_history_prompt_stem_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（一般出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=False)


def _coerce_exam_quiz_history_prompt_followup_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（追問出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=True)


_EXAM_QUIZ_HISTORY_LIST_FIELD = Field(
    default_factory=list,
    description="先前問答（八欄位 JSON 物件陣列）；僅寫入 DB",
)
_EXAM_QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD = Field(
    default_factory=list,
    description="併入 LLM 出題 prompt 的先前題幹（JSON 物件陣列，每筆僅 quiz_content）；寫入 DB",
)


class ExamQuizHistoryPromptStem(BaseModel):
    """quiz_history_list_prompt_text 單筆（一般出題）。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM]},
    )

    quiz_content: str = Field(..., description="先前題目題幹")


class ExamQuizHistoryPromptFollowup(BaseModel):
    """quiz_history_list_prompt_text 單筆（追問出題）。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM]},
    )

    quiz_content: str = Field(..., description="先前題目題幹")
    quiz_answer_reference: str = Field("", description="參考答案全文")
    answer_content: str = Field(
        "",
        description="學生先前作答",
        validation_alias=AliasChoices("answer_content", "quiz_answer", "answer"),
    )
    answer_critique: str = Field("", description="批改評語")


class ExamCreateLlmGenerateQuizRequest(BaseModel):
    """POST /exam/page/quiz/create-llm-generate；先 create 再 llm-generate，不需傳 exam_quiz_id。"""

    exam_page_id: str = Field("", description="目標 Exam 的 exam_page_id")
    rag_page_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_page_id（與 POST /rag/page/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    quiz_history_list: list[ExamQuizHistoryPair] = _EXAM_QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[ExamQuizHistoryPromptStem] = (
        _EXAM_QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_prompt_stem_validator(v)


class ExamLlmGenerateQuizRequest(BaseModel):
    """POST /exam/page/quiz/llm-generate；欄位順序同 public.Exam_Quiz（exam_quiz_id, rag_page_id, rag_unit_id, rag_quiz_id, quiz_history_list）。"""

    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵")
    rag_page_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_page_id（與 POST /rag/page/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz。列鎖鍵規則同 rag_unit_id",
    )
    quiz_history_list: list[ExamQuizHistoryPair] = _EXAM_QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[ExamQuizHistoryPromptStem] = (
        _EXAM_QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_prompt_stem_validator(v)


class ExamCreateLlmGenerateQuizFollowupRequest(BaseModel):
    """POST /exam/page/quiz/create-llm-generate-followup；先 create 再 llm-generate-followup，不需傳 exam_quiz_id。"""

    model_config = ConfigDict(populate_by_name=True)

    exam_page_id: str = Field("", description="目標 Exam 的 exam_page_id")
    rag_page_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_page_id（與 POST /rag/page/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    follow_up_exam_quiz_id: int = Field(
        0,
        ge=0,
        description="前一筆 Exam_Quiz 主鍵；>0 時寫入本列 follow_up=true 與此 id。傳 0 視為第一題（一般出題）",
        validation_alias=AliasChoices("follow_up_exam_quiz_id", "followUpExamQuizId"),
    )
    quiz_history_list: list[ExamQuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答（八欄位 JSON 物件陣列）；僅寫入 DB",
        validation_alias=AliasChoices("quiz_history_list", "quizHistoryList"),
    )
    quiz_history_list_prompt_text: list[ExamQuizHistoryPromptFollowup] = Field(
        default_factory=list,
        description=(
            "併入 LLM 追問 prompt 的先前問答（JSON 物件陣列：quiz_content、"
            "quiz_answer_reference、answer_content、answer_critique）；寫入 DB"
        ),
        validation_alias=AliasChoices(
            "quiz_history_list_prompt_text",
            "quizHistoryListPromptText",
        ),
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_prompt_followup_validator(v)


class ExamLlmGenerateQuizFollowupRequest(BaseModel):
    """POST /exam/page/quiz/llm-generate-followup；欄位順序同 Exam_Quiz 至 rag_quiz_id，接 follow_up_exam_quiz_id 與 quiz_history_list。"""

    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（本筆接續題）")
    rag_page_id: str = Field(
        ...,
        min_length=1,
        description="Rag.rag_page_id（與 POST /rag/page/create 等相同之 tab 識別字串）",
    )
    rag_unit_id: int = Field(
        ...,
        gt=0,
        description="Rag_Unit 主鍵（>0）。列尚未寫入時以此綁定；列已寫入須完全一致",
    )
    rag_quiz_id: int = Field(
        ...,
        gt=0,
        description="Rag_Quiz 主鍵（>0）；出題／作答模板 prompt 由此列讀取並於成功後寫入 Exam_Quiz",
    )
    follow_up_exam_quiz_id: int = Field(
        0,
        ge=0,
        description="前一筆 Exam_Quiz 主鍵；>0 時寫入本列 follow_up=true 與此 id。傳 0 視為第一題（一般出題）",
        validation_alias=AliasChoices("follow_up_exam_quiz_id", "followUpExamQuizId"),
    )
    quiz_history_list: list[ExamQuizHistoryPair] = Field(
        default_factory=list,
        description="先前問答（八欄位 JSON 物件陣列）；僅寫入 DB",
        validation_alias=AliasChoices("quiz_history_list", "quizHistoryList"),
    )
    quiz_history_list_prompt_text: list[ExamQuizHistoryPromptFollowup] = Field(
        default_factory=list,
        description=(
            "併入 LLM 追問 prompt 的先前問答（JSON 物件陣列：quiz_content、"
            "quiz_answer_reference、answer_content、answer_critique）；寫入 DB"
        ),
        validation_alias=AliasChoices(
            "quiz_history_list_prompt_text",
            "quizHistoryListPromptText",
        ),
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_exam_quiz_history_prompt_followup_validator(v)


class ExamQuizRateRequest(BaseModel):
    """POST /exam/page/quiz/quiz-rate：更新 Exam_Quiz.quiz_rate。"""
    exam_quiz_id: int = Field(..., ge=1, description="Exam_Quiz 主鍵")
    quiz_rate: ExamQuizRateValue = Field(0, description="僅 -1、0、1")


class ExamQuizGradeRateRequest(BaseModel):
    """POST /exam/page/quiz/grade-rate：更新 Exam_Quiz.grade_rate。"""
    exam_quiz_id: int = Field(..., ge=1, description="Exam_Quiz 主鍵")
    grade_rate: ExamQuizRateValue = Field(0, description="僅 -1、0、1")


class ExamQuizGradeRequest(BaseModel):
    """POST /exam/page/quiz/llm-grade：body 欄位依序對應 public.Exam_Quiz 之 exam_quiz_id、quiz_content、answer_content（學生作答 quiz_answer）。
    批改用 quiz_user_prompt_text／answer_user_prompt_text 優先採 Exam_Quiz 列，缺則自 Rag_Quiz 補齊。"""
    exam_quiz_id: int = Field(..., gt=0, description="Exam_Quiz 主鍵（必填，>0）；置入評分 prompt")
    quiz_content: str = Field(
        "",
        description="選填；若空則使用 Exam_Quiz 中存的 quiz_content；置入評分 prompt",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答；寫入 Exam_Quiz.answer_content；置入評分 prompt",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


# ---------------------------------------------------------------------------
# GET / PUT /exam/llm_api_key
# ---------------------------------------------------------------------------


class ExamApiKeyResponse(BaseModel):
    """GET/PUT /exam/llm_api_key 回應（Course_Setting key=exam-api-key）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class PutExamApiKeyRequest(BaseModel):
    """PUT /exam/llm_api_key 的 body。"""

    api_key: str = Field(..., description="Exam LLM API Key")
