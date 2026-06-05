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
    """POST /rag/page/unit/quiz/llm-generate；含 quiz_history_list（八欄位物件陣列）。"""

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
    POST /rag/page/unit/quiz/llm-grade 請求 body。
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
    """POST /rag/page/unit/quiz/llm-generate-db；含 quiz_history_list（八欄位物件陣列）。"""

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
    """POST /rag/page/unit/quiz/llm-generate-followup；含 quiz_history_list（先前問答 JSON 陣列，對齊 DB 欄位）。"""

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
    """POST /rag/page/unit/quiz/llm-generate-followup-db；不含 quiz_user_prompt_text。"""

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
    POST /rag/page/unit/quiz/llm-grade-db。
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
    POST /rag/page/unit/quiz/for-exam：欄位順序同 public.Rag_Quiz（rag_quiz_id, rag_page_id, rag_unit_id, for_exam）。
    以 rag_quiz_id 更新 Rag_Quiz.for_exam；若一併傳入 rag_page_id／rag_unit_id（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_page_id: str = Field("", description="選填；與資料列 rag_page_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    for_exam: bool = Field(True, description="true：標記為測驗用；false：取消測驗用")


class RagQuizFollowupRequest(BaseModel):
    """
    POST /rag/page/unit/quiz/followup：欄位順序同 public.Rag_Quiz（rag_quiz_id, rag_page_id, rag_unit_id, follow_up）。
    以 rag_quiz_id 更新 Rag_Quiz.follow_up；若一併傳入 rag_page_id／rag_unit_id（>0），須與該列一致。
    """

    rag_quiz_id: int = Field(..., gt=0, description="Rag_Quiz 主鍵")
    rag_page_id: str = Field("", description="選填；與資料列 rag_page_id 須一致")
    rag_unit_id: int = Field(0, ge=0, description="選填；>0 時須與資料列 rag_unit_id 一致")
    followup: bool = Field(
        False,
        description="true：標記為追問題；false：取消追問標記",
        validation_alias=AliasChoices("followup", "follow_up", "followUp"),
    )


class RagUnitTextResponse(BaseModel):
    """GET /rag/unit/text 回應。"""

    rag_page_id: str
    folder_name: str = ""
    rag_unit_id: int = 0
    text_file_name: str = ""
    transcript: str = ""


class RagUnitMp3FileFromZipResponse(BaseModel):
    """GET /rag/unit/mp3-file：自 upload ZIP 擷取之音訊與同資料夾文字檔逐字稿。"""

    rag_page_id: str
    folder_name: str
    audio_base64: str
    media_type: str
    filename: str
    text_file_name: str = ""
    transcript: str = ""


class RagUnitYoutubeUrlFromZipResponse(BaseModel):
    """GET /rag/unit/youtube-url：自 upload ZIP 解析 watch URL 與文字檔第二行起逐字稿。"""

    rag_page_id: str
    folder_name: str
    youtube_url: str = Field(..., description="https://www.youtube.com/watch?v=…")
    text_file_name: str = ""
    transcript: str = ""


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm_api_key
# ---------------------------------------------------------------------------


class RagApiKeyResponse(BaseModel):
    """GET/PUT /rag/llm_api_key 回應（Course_Setting key=rag-api-key）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    api_key: Optional[str] = None


class PutRagApiKeyRequest(BaseModel):
    """PUT /rag/llm_api_key 的 body。"""

    api_key: str = Field(..., description="RAG LLM API Key")


# ---------------------------------------------------------------------------
# GET / PUT /rag/llm_model
# ---------------------------------------------------------------------------


class RagLlmModelResponse(BaseModel):
    """GET/PUT /rag/llm_model 回應（Course_Setting key=llm-model；出題、批改、弱點分析共用）。"""

    course_setting_id: Optional[int] = None
    course_id: int
    llm_model: Optional[str] = None


class PutRagLlmModelRequest(BaseModel):
    """PUT /rag/llm_model 的 body。"""

    llm_model: str = Field(
        ...,
        description="RAG 出題／批改／弱點分析 LLM 模型名（對應 QUIZ_LLM_MODEL／GRADE_LLM_MODEL／WEAKNESS_LLM_MODEL，預設 gpt-5.4）",
    )
