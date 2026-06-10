"""
LLM Prompt 模板查詢 API。
- GET /v1/prompt-templates：回傳抓 RAG、llm-generate、llm-answer、llm-ask、bank、quiz、個人分析、課程分析之 prompt 全文（程式內建模板，非 Course_Setting 或 DB 動態值）。
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId

from services.answering import (
    SYSTEM_PROMPT_ANSWER,
    USER_PROMPT_ANSWER_FAISS_COURSE,
    USER_PROMPT_ANSWER_TRANSCRIPT_COURSE,
)
from services.asking import (
    SYSTEM_PROMPT_ASK,
    USER_PROMPT_ASK_FAISS_COURSE,
    USER_PROMPT_ASK_TRANSCRIPT_COURSE,
)
from services.quiz_generation import (
    SYSTEM_PROMPT_QUIZ,
    SYSTEM_PROMPT_QUIZ_FOLLOWUP,
    USER_PROMPT_COURSE,
    USER_PROMPT_COURSE_FOLLOWUP,
)
from services.bank_generation import (
    SYSTEM_PROMPT_BANK_QUIZ,
    USER_PROMPT_BANK_COURSE,
)
from services.bank_answering import (
    SYSTEM_PROMPT_BANK_ANSWER,
    USER_PROMPT_BANK_ANSWER_FAISS_COURSE,
    USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE,
)
from services.quiz_asking import (
    SYSTEM_PROMPT_QUIZ_ASK,
    USER_PROMPT_QUIZ_ASK_FAISS_COURSE,
    USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE,
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


class LlmAskPrompts(BaseModel):
    """POST .../llm-ask 所用 prompt 模板（學生答題後追問課程內容）。"""

    system: str = Field(..., description="追問回答 system prompt（SYSTEM_PROMPT_ASK）")
    user_transcript_course: str = Field(
        ...,
        description="逐字稿路徑 user prompt（USER_PROMPT_ASK_TRANSCRIPT_COURSE）",
    )
    user_faiss_course: str = Field(
        ...,
        description="FAISS 檢索路徑 user prompt（USER_PROMPT_ASK_FAISS_COURSE）",
    )


class BankPrompts(BaseModel):
    """Bank（測試題庫）出題／批改 prompt 模板（bank 專屬，與 rag 無關；題組 question_system_prompt_text 會織入出題 system）。"""

    llm_generate_system: str = Field(..., description="出題 system（SYSTEM_PROMPT_BANK_QUIZ）")
    llm_generate_user: str = Field(..., description="出題 user（USER_PROMPT_BANK_COURSE）")
    llm_answer_system: str = Field(..., description="批改 system（SYSTEM_PROMPT_BANK_ANSWER）")
    llm_answer_user_transcript_course: str = Field(
        ..., description="批改 user（逐字稿路徑，USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE）"
    )
    llm_answer_user_faiss_course: str = Field(
        ..., description="批改 user（FAISS 路徑，USER_PROMPT_BANK_ANSWER_FAISS_COURSE）"
    )


class QuizPrompts(BaseModel):
    """Quiz（試卷／Test）prompt 模板。

    出題／批改沿用 `bank` 區塊（quiz 重用 bank 的 LLM 管線，金鑰／模型走 quiz- 設定）；
    此處為 quiz 專屬之「追問」prompt（學生對題組對應之 Bank 課程內容發問，services.quiz_asking）。
    """

    llm_ask_system: str = Field(..., description="追問回答 system（SYSTEM_PROMPT_QUIZ_ASK）")
    llm_ask_user_transcript_course: str = Field(
        ..., description="追問 user（逐字稿路徑，USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE）"
    )
    llm_ask_user_faiss_course: str = Field(
        ..., description="追問 user（FAISS 路徑，USER_PROMPT_QUIZ_ASK_FAISS_COURSE）"
    )


class AnalysisPrompts(PromptPair):
    """個人／課程弱點分析 LLM prompt 模板。"""


class PromptItem(BaseModel):
    """單一 prompt 全文（已標明 system／user 與變體）。"""

    role: str = Field(..., description="prompt 角色：system 或 user")
    variant: str = Field("", description="變體：空＝預設；followup＝追問；transcript_course／faiss_course＝批改/追問的逐字稿/向量檢索路徑")
    name: str = Field(..., description="程式內常數名（如 SYSTEM_PROMPT_BANK_QUIZ）")
    description: str = Field(..., description="此 prompt 的用途說明")
    content: str = Field(..., description="prompt 全文（`{占位符}` 保留原樣）")


class PromptSection(BaseModel):
    """模組內的用途分區（出題／批改／追問／分析）。"""

    section: str = Field(..., description="用途代碼：generate（出題）／answer（批改）／ask（追問）／analyze（分析）")
    label: str = Field(..., description="用途中文標籤")
    prompts: list[PromptItem] = Field(..., description="此用途下的所有 prompt（system／user）")


class PromptModule(BaseModel):
    """一個功能模組（RAG／Exam／Bank／Quiz／Person-Setting／Course-Setting）之 prompt。"""

    module: str = Field(..., description="模組代碼：rag／exam／bank／quiz／person_setting／course_setting")
    label: str = Field(..., description="模組中文標籤")
    description: str = Field("", description="此模組對應的端點與 prompt 來源說明")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="模組層級的非-prompt 屬性（如 RAG 向量檢索 retrieval_query／retrieval_k、建庫預設 embedding／chunk）",
    )
    sections: list[PromptSection] = Field(..., description="依用途分區（出題／出題-追問／批改／追問／分析）")


class AllPromptTemplatesResponse(BaseModel):
    """GET /v1/prompt-templates 回應。"""

    placeholders: dict[str, dict[str, str]] = Field(
        ...,
        description="各區塊模板內 `{占位符}` 之填入來源與意義",
    )
    # 依模組分類、含全文的完整清單（前端逐模組／逐用途顯示用）
    modules: list[PromptModule] = Field(
        ...,
        description="所有 prompt 全文，依模組分類（rag／exam／bank／quiz／person_setting／course_setting），每模組再依用途（出題／批改／追問／分析）與 system／user 細分",
    )
    # 以下為原扁平欄位（保留相容）
    rag: RagPrompts
    llm_generate: LlmGeneratePrompts
    llm_answer: LlmAnswerPrompts
    llm_ask: LlmAskPrompts
    bank: BankPrompts
    quiz: QuizPrompts
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

    # 共用的用途分區（RAG 與 Exam 出題／批改 prompt 相同；Bank 與 Quiz 出題／批改 prompt 相同）
    # 出題分「一般（無 followup）」與「追問（followup）」兩個 section。
    def _rag_generate_section() -> PromptSection:
        return PromptSection(section="generate", label="出題（一般，無 followup）", prompts=[
            PromptItem(role="system", variant="", name="SYSTEM_PROMPT_QUIZ", description="一般出題 system", content=SYSTEM_PROMPT_QUIZ),
            PromptItem(role="user", variant="", name="USER_PROMPT_COURSE", description="一般出題 user", content=USER_PROMPT_COURSE),
        ])

    def _rag_generate_followup_section() -> PromptSection:
        return PromptSection(section="generate_followup", label="出題（追問 followup）", prompts=[
            PromptItem(role="system", variant="followup", name="SYSTEM_PROMPT_QUIZ_FOLLOWUP", description="追問出題 system", content=SYSTEM_PROMPT_QUIZ_FOLLOWUP),
            PromptItem(role="user", variant="followup", name="USER_PROMPT_COURSE_FOLLOWUP", description="追問出題 user", content=USER_PROMPT_COURSE_FOLLOWUP),
        ])

    def _rag_answer_section() -> PromptSection:
        return PromptSection(section="answer", label="批改", prompts=[
            PromptItem(role="system", variant="", name="SYSTEM_PROMPT_ANSWER", description="批改 system", content=SYSTEM_PROMPT_ANSWER),
            PromptItem(role="user", variant="transcript_course", name="USER_PROMPT_ANSWER_TRANSCRIPT_COURSE", description="批改 user（逐字稿路徑）", content=USER_PROMPT_ANSWER_TRANSCRIPT_COURSE),
            PromptItem(role="user", variant="faiss_course", name="USER_PROMPT_ANSWER_FAISS_COURSE", description="批改 user（向量檢索路徑）", content=USER_PROMPT_ANSWER_FAISS_COURSE),
        ])

    def _bank_generate_section() -> PromptSection:
        return PromptSection(section="generate", label="出題", prompts=[
            PromptItem(role="system", variant="", name="SYSTEM_PROMPT_BANK_QUIZ", description="出題 system（題組 question_system_prompt_text 會織入其後）", content=SYSTEM_PROMPT_BANK_QUIZ),
            PromptItem(role="user", variant="", name="USER_PROMPT_BANK_COURSE", description="出題 user", content=USER_PROMPT_BANK_COURSE),
        ])

    def _bank_answer_section() -> PromptSection:
        return PromptSection(section="answer", label="批改", prompts=[
            PromptItem(role="system", variant="", name="SYSTEM_PROMPT_BANK_ANSWER", description="批改 system", content=SYSTEM_PROMPT_BANK_ANSWER),
            PromptItem(role="user", variant="transcript_course", name="USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE", description="批改 user（逐字稿路徑）", content=USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE),
            PromptItem(role="user", variant="faiss_course", name="USER_PROMPT_BANK_ANSWER_FAISS_COURSE", description="批改 user（向量檢索路徑）", content=USER_PROMPT_BANK_ANSWER_FAISS_COURSE),
        ])

    # 依模組分類（RAG／Exam／Bank／Quiz／Person-Setting／Course-Setting）
    modules = [
        PromptModule(
            module="rag",
            label="RAG",
            description="POST /v1/rag/quizzes/llm-generate（及 followup）、llm-answer",
            attributes={
                "retrieval": {
                    "llm_generate": rag_cfg["llm_generate"],
                    "llm_answer": rag_cfg["llm_answer"],
                },
                "build_defaults": rag_build_defaults(),
            },
            sections=[_rag_generate_section(), _rag_generate_followup_section(), _rag_answer_section()],
        ),
        PromptModule(
            module="exam",
            label="Exam",
            description="POST /v1/exam/quizzes/llm-generate、llm-answer、llm-ask（出題／批改 prompt 與檢索屬性與 RAG 共用）",
            attributes={
                "retrieval": {
                    "llm_generate": rag_cfg["llm_generate"],
                    "llm_answer": rag_cfg["llm_answer"],
                },
                "build_defaults": rag_build_defaults(),
            },
            sections=[
                _rag_generate_section(),
                _rag_generate_followup_section(),
                _rag_answer_section(),
                PromptSection(section="ask", label="追問", prompts=[
                    PromptItem(role="system", variant="", name="SYSTEM_PROMPT_ASK", description="追問回答 system", content=SYSTEM_PROMPT_ASK),
                    PromptItem(role="user", variant="transcript_course", name="USER_PROMPT_ASK_TRANSCRIPT_COURSE", description="追問 user（逐字稿路徑）", content=USER_PROMPT_ASK_TRANSCRIPT_COURSE),
                    PromptItem(role="user", variant="faiss_course", name="USER_PROMPT_ASK_FAISS_COURSE", description="追問 user（向量檢索路徑）", content=USER_PROMPT_ASK_FAISS_COURSE),
                ]),
            ],
        ),
        PromptModule(
            module="bank",
            label="Bank",
            description="POST /v1/bank/groups/{id}/qa/llm-generate、POST /v1/bank/qa/{id}/llm-answer",
            sections=[_bank_generate_section(), _bank_answer_section()],
        ),
        PromptModule(
            module="quiz",
            label="Quiz",
            description="POST /v1/quiz/groups/{id}/qa/llm-generate、llm-answer（出題／批改 prompt 與 Bank 共用）、llm-ask",
            sections=[
                _bank_generate_section(),
                _bank_answer_section(),
                PromptSection(section="ask", label="追問", prompts=[
                    PromptItem(role="system", variant="", name="SYSTEM_PROMPT_QUIZ_ASK", description="追問回答 system", content=SYSTEM_PROMPT_QUIZ_ASK),
                    PromptItem(role="user", variant="transcript_course", name="USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE", description="追問 user（逐字稿路徑）", content=USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE),
                    PromptItem(role="user", variant="faiss_course", name="USER_PROMPT_QUIZ_ASK_FAISS_COURSE", description="追問 user（向量檢索路徑）", content=USER_PROMPT_QUIZ_ASK_FAISS_COURSE),
                ]),
            ],
        ),
        PromptModule(
            module="person_setting",
            label="Person-Setting（個人弱點分析）",
            description="POST /v1/person-analyses/{id}/llm-analysis",
            sections=[
                PromptSection(section="analyze", label="分析", prompts=[
                    PromptItem(role="system", variant="", name="PERSON_ANALYSIS_SYSTEM", description="個人分析 system", content=person_tpl["system"]),
                    PromptItem(role="user", variant="", name="PERSON_ANALYSIS_USER", description="個人分析 user", content=person_tpl["user"]),
                ]),
            ],
        ),
        PromptModule(
            module="course_setting",
            label="Course-Setting（課程弱點分析）",
            description="POST /v1/course-analyses/{id}/llm-analysis",
            sections=[
                PromptSection(section="analyze", label="分析", prompts=[
                    PromptItem(role="system", variant="", name="COURSE_ANALYSIS_SYSTEM", description="課程分析 system", content=course_tpl["system"]),
                    PromptItem(role="user", variant="", name="COURSE_ANALYSIS_USER", description="課程分析 user", content=course_tpl["user"]),
                ]),
            ],
        ),
    ]

    return AllPromptTemplatesResponse(
        placeholders=prompt_placeholder_descriptions(),
        modules=modules,
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
        llm_ask=LlmAskPrompts(
            system=SYSTEM_PROMPT_ASK,
            user_transcript_course=USER_PROMPT_ASK_TRANSCRIPT_COURSE,
            user_faiss_course=USER_PROMPT_ASK_FAISS_COURSE,
        ),
        bank=BankPrompts(
            llm_generate_system=SYSTEM_PROMPT_BANK_QUIZ,
            llm_generate_user=USER_PROMPT_BANK_COURSE,
            llm_answer_system=SYSTEM_PROMPT_BANK_ANSWER,
            llm_answer_user_transcript_course=USER_PROMPT_BANK_ANSWER_TRANSCRIPT_COURSE,
            llm_answer_user_faiss_course=USER_PROMPT_BANK_ANSWER_FAISS_COURSE,
        ),
        quiz=QuizPrompts(
            llm_ask_system=SYSTEM_PROMPT_QUIZ_ASK,
            llm_ask_user_transcript_course=USER_PROMPT_QUIZ_ASK_TRANSCRIPT_COURSE,
            llm_ask_user_faiss_course=USER_PROMPT_QUIZ_ASK_FAISS_COURSE,
        ),
        person_analysis=AnalysisPrompts(**person_tpl),
        course_analysis=AnalysisPrompts(**course_tpl),
    )
