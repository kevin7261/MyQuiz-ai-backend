"""routers.grade schemas（自 grade.py 拆分）。"""

from typing import Any, Optional


from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from utils.db_schema import (
    QUIZ_HISTORY_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_FOLLOWUP_OPENAPI_ITEM,
    QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM,
    coerce_quiz_history_prompt_text_request,
    coerce_quiz_history_request,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QuizHistoryPair(BaseModel):
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


def _coerce_quiz_history_list_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list（不讀 DB）。"""
    return coerce_quiz_history_request(v)


def _coerce_quiz_history_prompt_stem_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（一般出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=False)


def _coerce_quiz_history_prompt_followup_validator(v: Any) -> Any:
    """正規化 API 傳入的 quiz_history_list_prompt_text（追問出題）。"""
    return coerce_quiz_history_prompt_text_request(v, followup=True)


_QUIZ_HISTORY_LIST_FIELD = Field(
    default_factory=list,
    description="先前問答（八欄位 JSON 物件陣列）；僅寫入 DB",
)
_QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD = Field(
    default_factory=list,
    description="併入 LLM 出題 prompt 的先前題幹（JSON 物件陣列，每筆僅 quiz_content）；寫入 DB",
)
_QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD = Field(
    default_factory=list,
    description=(
        "併入 LLM 追問 prompt 的先前問答（JSON 物件陣列：quiz_content、"
        "quiz_answer_reference、answer_content、answer_critique）；寫入 DB"
    ),
)


class QuizHistoryPromptStem(BaseModel):
    """quiz_history_list_prompt_text 單筆（一般出題）。"""

    model_config = ConfigDict(
        json_schema_extra={"examples": [QUIZ_HISTORY_PROMPT_STEM_OPENAPI_ITEM]},
    )

    quiz_content: str = Field(..., description="先前題目題幹")


class QuizHistoryPromptFollowup(BaseModel):
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


class GenerateQuizRequest(BaseModel):
    """POST /rag/quizzes/llm-generate；含 quiz_history_list（八欄位物件陣列）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptStem] = _QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_stem_validator(v)


class QuizGradeRequest(BaseModel):
    """
    POST /rag/quizzes/llm-grade 請求 body。
    欄位順序：Rag.rag_id → public.Rag_Quiz（rag_page_id, rag_quiz_id, quiz_content, answer_user_prompt_text, answer_content／quiz_answer）。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_page_id: str = Field("", description="選填；後端以 Rag.rag_page_id 為準")
    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
    quiz_content: str = Field(
        "",
        description="選填；非空時優先並可寫入 Rag_Quiz；空則自該 rag_quiz_id 之 Rag_Quiz.quiz_content 讀取（須於庫記憶中有題幹）",
    )
    answer_user_prompt_text: str = Field(
        "",
        description="作答／批改 user prompt（可空）；寫入 Rag_Quiz.answer_user_prompt_text 並供評分 prompt 參考",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Quiz.answer_content）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


class GenerateQuizDbOnlyRequest(BaseModel):
    """POST /rag/quizzes/llm-generate-db；含 quiz_history_list（八欄位物件陣列）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptStem] = _QUIZ_HISTORY_LIST_PROMPT_STEM_FIELD

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_stem_validator(v)


class GenerateQuizFollowupRequest(BaseModel):
    """POST /rag/quizzes/llm-generate-followup；含 quiz_history_list（先前問答 JSON 陣列，對齊 DB 欄位）。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_user_prompt_text: str = Field(
        "",
        description="出題 user prompt（可空）；非空時優先並寫入 Rag_Quiz；空則自該列 Rag_Quiz.quiz_user_prompt_text 帶入 LLM",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptFollowup] = (
        _QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_followup_validator(v)


class GenerateQuizFollowupDbOnlyRequest(BaseModel):
    """POST /rag/quizzes/llm-generate-followup-db；不含 quiz_user_prompt_text。"""

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    quiz_name: str = Field(
        "",
        description="測驗名稱；空字串則由 repack stem／單元 unit_name 決定 Rag_Quiz.quiz_name",
    )
    quiz_history_list: list[QuizHistoryPair] = _QUIZ_HISTORY_LIST_FIELD
    quiz_history_list_prompt_text: list[QuizHistoryPromptFollowup] = (
        _QUIZ_HISTORY_LIST_PROMPT_FOLLOWUP_FIELD
    )

    @field_validator("quiz_history_list", mode="before")
    @classmethod
    def _coerce_quiz_history_list(cls, v: Any) -> Any:
        return _coerce_quiz_history_list_validator(v)

    @field_validator("quiz_history_list_prompt_text", mode="before")
    @classmethod
    def _coerce_quiz_history_list_prompt_text(cls, v: Any) -> Any:
        return _coerce_quiz_history_prompt_followup_validator(v)


class QuizGradeDbOnlyRequest(BaseModel):
    """
    POST /rag/quizzes/llm-grade-db。
    欄位順序：Rag.rag_id → Rag_Quiz（rag_page_id, rag_quiz_id, quiz_content, answer_content／quiz_answer）；不含 answer_user_prompt_text。
    """

    rag_id: str = Field(
        "",
        description="必填；Rag 表 rag_id（字串）；載入講義／向量 ZIP 並驗證存取權",
    )
    rag_page_id: str = Field("", description="選填；後端以 Rag.rag_page_id 為準")
    rag_quiz_id: str = Field("", description="必填（數字字串 >0）；Rag_Quiz 主鍵")
    quiz_content: str = Field(
        "",
        description="選填；非空時優先並可寫入 Rag_Quiz；空則自該 rag_quiz_id 之 Rag_Quiz.quiz_content 讀取（須於庫記憶中有題幹）",
    )
    quiz_answer: str = Field(
        ...,
        description="學生作答（寫入 Rag_Quiz.answer_content）；相容舊 JSON 欄位 answer",
        validation_alias=AliasChoices("quiz_answer", "answer"),
    )


class RagQuizForExamRequest(BaseModel):
    """
    PUT /rag/quizzes/{rag_quiz_id}/for-exam：以 path 之 rag_quiz_id 更新 Rag_Quiz.for_exam。
    body 僅含旗標值 for_exam。
    """

    for_exam: bool = Field(True, description="true：標記為測驗用；false：取消測驗用")


class RagQuizFollowupRequest(BaseModel):
    """
    PUT /rag/quizzes/{rag_quiz_id}/followup：以 path 之 rag_quiz_id 更新 Rag_Quiz.follow_up。
    body 僅含旗標值 followup。
    """

    followup: bool = Field(
        False,
        description="true：標記為追問題；false：取消追問標記",
        validation_alias=AliasChoices("followup", "follow_up", "followUp"),
    )


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm-api-key
# ---------------------------------------------------------------------------


class RagApiKeyResponse(BaseModel):
    """GET/PUT /rag/llm-api-key 回應（Course_Setting key=rag-api-key）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class PutRagApiKeyRequest(BaseModel):
    """PUT /rag/llm-api-key 的 body。"""

    api_key: str = Field(..., description="RAG LLM API Key")


class RagApiKeyExistsResponse(BaseModel):
    """GET /rag/llm-api-key/exists 回應。"""

    course_id: int
    exists: bool = Field(..., description="該課程是否已設定非空 rag-api-key")


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm-model
# ---------------------------------------------------------------------------


class RagLlmModelResponse(BaseModel):
    """GET/PUT /rag/llm-model 回應（Course_Setting key=llm-model；出題、批改、弱點分析共用）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutRagLlmModelRequest(BaseModel):
    """PUT /rag/llm-model 的 body。"""

    llm_model: str = Field(
        ...,
        description="RAG 出題／批改／弱點分析 LLM 模型名（對應 QUIZ_LLM_MODEL／GRADE_LLM_MODEL／WEAKNESS_LLM_MODEL，預設 gpt-5.4）",
    )
