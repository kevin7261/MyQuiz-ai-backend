"""
LLM Prompt 模板查詢 API。
- GET /prompt/templates：回傳抓 RAG、llm-generate、llm-grade、個人分析、課程分析之 prompt 全文（程式內建模板，非 System_Setting 或 DB 動態值）。
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId

from services.grading import (
    SYSTEM_PROMPT_GRADE,
    USER_PROMPT_GRADE_FAISS_COURSE,
    USER_PROMPT_GRADE_TRANSCRIPTION_COURSE,
)
from services.quiz_generation import (
    SYSTEM_PROMPT_QUIZ,
    SYSTEM_PROMPT_QUIZ_FOLLOWUP,
    USER_PROMPT_COURSE,
    USER_PROMPT_COURSE_FOLLOWUP,
)
from services.rag_prompts import rag_build_defaults, rag_prompt_templates
from services.weakness_report import (
    analysis_prompt_templates,
    COURSE_ANALYSIS_LABEL,
    PERSON_ANALYSIS_LABEL,
)

router = APIRouter(prefix="/prompt", tags=["prompt"])


class PromptPair(BaseModel):
    system: str
    user: str


class RagPrompts(BaseModel):
    """unit_type=1 自 FAISS 抓 RAG 片段之向量檢索查詢（非 Chat LLM 模板；system 固定空字串）。"""

    llm_generate: PromptPair = Field(
        ...,
        description="llm-generate 檢索句：user 為固定查詢「課程重點概念」",
    )
    llm_grade: PromptPair = Field(
        ...,
        description="llm-grade 檢索句：user 占位 {quiz_content}，實際以題幹代入",
    )
    build_defaults: dict[str, int | str] = Field(
        ...,
        description="build-rag-zip 建 FAISS 預設 embedding／chunk／retrieval_k",
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


class LlmGradePrompts(BaseModel):
    """POST .../llm-grade 所用 prompt 模板。"""

    system: str = Field(..., description="批改 system prompt（SYSTEM_PROMPT_GRADE）")
    user_transcription_course: str = Field(
        ...,
        description="逐字稿路徑 user prompt（USER_PROMPT_GRADE_TRANSCRIPTION_COURSE）",
    )
    user_faiss_course: str = Field(
        ...,
        description="FAISS 檢索路徑 user prompt（USER_PROMPT_GRADE_FAISS_COURSE）",
    )


class AnalysisPrompts(PromptPair):
    """個人／課程弱點分析 LLM prompt 模板。"""


class AllPromptTemplatesResponse(BaseModel):
    """GET /prompt/templates 回應。"""

    rag: RagPrompts
    llm_generate: LlmGeneratePrompts
    llm_grade: LlmGradePrompts
    person_analysis: AnalysisPrompts
    course_analysis: AnalysisPrompts


@router.get("/templates", response_model=AllPromptTemplatesResponse)
def get_all_prompt_templates(_person_id: PersonId):
    """
    回傳各 LLM 功能之 prompt 模板全文。
    占位符（如 `{context_md}`、`{quiz_user_prompt_text}`）保留原樣，供前端或文件對照。
    `rag` 區塊為向量檢索抓 RAG 片段之查詢句（非 Chat LLM system 模板）。
    """
    person_tpl = analysis_prompt_templates(PERSON_ANALYSIS_LABEL)
    course_tpl = analysis_prompt_templates(COURSE_ANALYSIS_LABEL)
    rag_tpl = rag_prompt_templates()
    return AllPromptTemplatesResponse(
        rag=RagPrompts(
            llm_generate=PromptPair(**rag_tpl["llm_generate"]),
            llm_grade=PromptPair(**rag_tpl["llm_grade"]),
            build_defaults=rag_build_defaults(),
        ),
        llm_generate=LlmGeneratePrompts(
            system=SYSTEM_PROMPT_QUIZ,
            user=USER_PROMPT_COURSE,
            system_followup=SYSTEM_PROMPT_QUIZ_FOLLOWUP,
            user_followup=USER_PROMPT_COURSE_FOLLOWUP,
        ),
        llm_grade=LlmGradePrompts(
            system=SYSTEM_PROMPT_GRADE,
            user_transcription_course=USER_PROMPT_GRADE_TRANSCRIPTION_COURSE,
            user_faiss_course=USER_PROMPT_GRADE_FAISS_COURSE,
        ),
        person_analysis=AnalysisPrompts(**person_tpl),
        course_analysis=AnalysisPrompts(**course_tpl),
    )
