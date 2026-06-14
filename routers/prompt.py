"""
LLM Prompt 模板查詢 API。
- GET /v1/prompt-templates：回傳 bank、quiz 之 prompt 全文（程式內建模板，非 Course_Setting 或 DB 動態值）。
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dependencies.person_id import PersonId

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

router = APIRouter(tags=["prompt"])


class PromptPair(BaseModel):
    system: str
    user: str


class BankPrompts(BaseModel):
    """Bank（測試題庫）出題／批改 prompt 模板（題組 question_system_prompt_text 會織入出題 system）。"""

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
    """Quiz（測驗／Test）prompt 模板。

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


class PromptItem(BaseModel):
    """單一 prompt 全文（已標明 system／user 與變體）。"""

    role: str = Field(..., description="prompt 角色：system 或 user")
    variant: str = Field("", description="變體：空＝預設；transcript_course／faiss_course＝批改/追問的逐字稿/向量檢索路徑")
    name: str = Field(..., description="程式內常數名（如 SYSTEM_PROMPT_BANK_QUIZ）")
    description: str = Field(..., description="此 prompt 的用途說明")
    content: str = Field(..., description="prompt 全文（`{占位符}` 保留原樣）")


class PromptSection(BaseModel):
    """模組內的用途分區（出題／批改／追問）。"""

    section: str = Field(..., description="用途代碼：generate（出題）／answer（批改）／ask（追問）")
    label: str = Field(..., description="用途中文標籤")
    prompts: list[PromptItem] = Field(..., description="此用途下的所有 prompt（system／user）")


class PromptModule(BaseModel):
    """一個功能模組（Bank／Quiz）之 prompt。"""

    module: str = Field(..., description="模組代碼：bank／quiz")
    label: str = Field(..., description="模組中文標籤")
    description: str = Field("", description="此模組對應的端點與 prompt 來源說明")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="模組層級的非-prompt 屬性",
    )
    sections: list[PromptSection] = Field(..., description="依用途分區（出題／批改／追問）")


class AllPromptTemplatesResponse(BaseModel):
    """GET /v1/prompt-templates 回應。"""

    placeholders: dict[str, dict[str, str]] = Field(
        ...,
        description="各區塊模板內 `{占位符}` 之填入來源與意義",
    )
    # 依模組分類、含全文的完整清單（前端逐模組／逐用途顯示用）
    modules: list[PromptModule] = Field(
        ...,
        description="所有 prompt 全文，依模組分類（bank／quiz），每模組再依用途（出題／批改／追問）與 system／user 細分",
    )
    # 以下為原扁平欄位（保留相容）
    bank: BankPrompts
    quiz: QuizPrompts


@router.get("/prompt-templates", response_model=AllPromptTemplatesResponse)
def get_all_prompt_templates(_person_id: PersonId):
    """
    回傳各 LLM 功能之 prompt 模板全文。
    模板內 `{占位符}` 保留原樣；`placeholders` 說明各占位符填入內容。
    """

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

    # 依模組分類（Bank／Quiz）
    modules = [
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
    ]

    return AllPromptTemplatesResponse(
        placeholders=prompt_placeholder_descriptions(),
        modules=modules,
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
    )
