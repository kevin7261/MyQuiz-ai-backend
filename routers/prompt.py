"""
LLM Prompt 模板查詢 API。
- GET /v1/prompt-templates：回傳抓 RAG、llm-generate、llm-answer、個人分析、課程分析之 prompt 全文（程式內建模板，非 Course_Setting 或 DB 動態值）。
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId

from services.answering import (
    SYSTEM_PROMPT_ANSWER,
    USER_PROMPT_ANSWER_FAISS_COURSE,
    USER_PROMPT_ANSWER_TRANSCRIPT_COURSE,
)
from services.quiz_generation import (
    SYSTEM_PROMPT_QUIZ,
    SYSTEM_PROMPT_QUIZ_FOLLOWUP,
    USER_PROMPT_COURSE,
    USER_PROMPT_COURSE_FOLLOWUP,
)
from services.prompt_placeholders import prompt_placeholder_descriptions
from services.rag_prompts import rag_build_defaults, rag_retrieval_config
from services.weakness_report import (
    analysis_prompt_templates,
    COURSE_ANALYSIS_LABEL,
    PERSON_ANALYSIS_LABEL,
)

router = APIRouter(tags=["prompt"])


class PromptPair(BaseModel):
    system: str
    user: str


class RagRetrieval(BaseModel):
    """unit_type=1 向量檢索設定（非 Chat LLM prompt）。"""

    retrieval_query: str = Field(..., description="送進 retriever 的查詢句")
    retrieval_k: int = Field(..., description="檢索回傳 chunk 數")


class RagPrompts(BaseModel):
    """unit_type=1 自 FAISS 抓 RAG 片段之設定（非 Chat LLM system／user 模板）。"""

    llm_generate: RagRetrieval
    llm_answer: RagRetrieval
    build_defaults: dict[str, int | str] = Field(
        ...,
        description="build-rag-zip 建 FAISS 預設 embedding／chunk",
    )


class LlmGeneratePrompts(BaseModel):
    """POST .../llm-generate（及 followup）所用 prompt 模板。"""

    system: str = Field(..., description="一般出題 system prompt（SYSTEM_PROMPT_QUIZ）")
    user: str = Field(..., description="一般出題 user prompt（USER_PROMPT_COURSE）")
    system_followup: str = Field(
        ..., description="追問出題 system prompt（SYSTEM_PROMPT_QUIZ_FOLLOWUP）"
    )
    user_followup: str = Field(
        ..., description="追問出題 user prompt（USER_PROMPT_COURSE_FOLLOWUP）"
    )


class LlmAnswerPrompts(BaseModel):
    """POST .../llm-answer 所用 prompt 模板。"""

    system: str = Field(..., description="批改 system prompt（SYSTEM_PROMPT_ANSWER）")
    user_transcript_course: str = Field(
        ...,
        description="逐字稿路徑 user prompt（USER_PROMPT_ANSWER_TRANSCRIPT_COURSE）",
    )
    user_faiss_course: str = Field(
        ...,
        description="FAISS 檢索路徑 user prompt（USER_PROMPT_ANSWER_FAISS_COURSE）",
    )


class AnalysisPrompts(PromptPair):
    """個人／課程弱點分析 LLM prompt 模板。"""


class AllPromptTemplatesResponse(BaseModel):
    """GET /v1/prompt-templates 回應。"""

    placeholders: dict[str, dict[str, str]] = Field(
        ...,
        description="各區塊模板內 `{占位符}` 之填入來源與意義",
    )
    rag: RagPrompts
    llm_generate: LlmGeneratePrompts
    llm_answer: LlmAnswerPrompts
    person_analysis: AnalysisPrompts
    course_analysis: AnalysisPrompts


@router.get("/prompt-templates", response_model=AllPromptTemplatesResponse)
def get_all_prompt_templates(_person_id: PersonId):
    """
    回傳各 LLM 功能之 prompt 模板全文。
    模板內 `{占位符}` 保留原樣；`placeholders` 說明各占位符填入內容。
    `rag` 區塊為向量檢索查詢句與 k 值（非 Chat LLM prompt）。
    """
    person_tpl = analysis_prompt_templates(PERSON_ANALYSIS_LABEL)
    course_tpl = analysis_prompt_templates(COURSE_ANALYSIS_LABEL)
    rag_cfg = rag_retrieval_config()
    return AllPromptTemplatesResponse(
        placeholders=prompt_placeholder_descriptions(),
        rag=RagPrompts(
            llm_generate=RagRetrieval(**rag_cfg["llm_generate"]),
            llm_answer=RagRetrieval(**rag_cfg["llm_answer"]),
            build_defaults=rag_build_defaults(),
        ),
        llm_generate=LlmGeneratePrompts(
            system=SYSTEM_PROMPT_QUIZ,
            user=USER_PROMPT_COURSE,
            system_followup=SYSTEM_PROMPT_QUIZ_FOLLOWUP,
            user_followup=USER_PROMPT_COURSE_FOLLOWUP,
        ),
        llm_answer=LlmAnswerPrompts(
            system=SYSTEM_PROMPT_ANSWER,
            user_transcript_course=USER_PROMPT_ANSWER_TRANSCRIPT_COURSE,
            user_faiss_course=USER_PROMPT_ANSWER_FAISS_COURSE,
        ),
        person_analysis=AnalysisPrompts(**person_tpl),
        course_analysis=AnalysisPrompts(**course_tpl),
    )
