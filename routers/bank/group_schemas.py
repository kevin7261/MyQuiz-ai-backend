"""routers.bank 題組（Bank_Group）／問答（Bank_QA）schemas。

對應 rag 的 Rag_Quiz LLM 出題／批改，但以「題組」為單位：
- 一個 Bank_Group 先設定 qa_count（本題組要出幾題）與出題／批改 prompt。
- question_system_prompt_text：連續出題的規定（如越來越難、勿重複），織入出題 system prompt（最高優先）。
- question_user_prompt_text：出題規定（出題 user prompt）。
- answer_user_prompt_text：批改規定（批改 user prompt）。
- 之後逐題產生 Bank_QA（每次一題），上限為 qa_count；無「追問」概念。
"""

from typing import Optional

from pydantic import BaseModel, Field

from utils.qa_count import QA_COUNT_DEFAULT, QA_COUNT_MAX, QA_COUNT_MIN


class CreateBankGroupRequest(BaseModel):
    """
    POST /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups：新增一個測試題組（Bank_Group），**不呼叫 LLM**。
    所屬單元由路徑 bank_page_id／bank_unit_id 決定。
    """

    group_name: str = Field("", description="題組顯示名稱")
    qa_count: int = Field(
        QA_COUNT_DEFAULT,
        ge=QA_COUNT_MIN,
        le=QA_COUNT_MAX,
        description="本題組預定要出的題數上限（1–20）；逐題產生時不會超過此數",
    )
    question_system_prompt_text: str = Field(
        "", description="連續出題的規定（織入出題 system prompt，最高優先；如越來越難、勿重複）"
    )
    question_user_prompt_text: str = Field("", description="出題規定（出題 user prompt）")
    question_llm_model: str = Field("", description="出題 LLM 模型；空則用課程 bank-llm-model 設定")
    answer_user_prompt_text: str = Field("", description="批改規定（批改 user prompt）")
    answer_llm_model: str = Field("", description="批改 LLM 模型；空則用課程 bank-llm-model 設定")
    for_exam: bool = Field(False, description="是否標記為測驗用")


class UpdateBankGroupRequest(BaseModel):
    """PATCH /bank/groups/{bank_group_id}：更新題組設定（僅更新有傳入的欄位）。"""

    group_name: Optional[str] = Field(None, description="新的題組顯示名稱")
    qa_count: Optional[int] = Field(
        None, ge=QA_COUNT_MIN, le=QA_COUNT_MAX, description="新的題數上限（1–20）"
    )
    question_system_prompt_text: Optional[str] = Field(None, description="新的連續出題規定")
    question_user_prompt_text: Optional[str] = Field(None, description="新的出題 user prompt")
    question_llm_model: Optional[str] = Field(None, description="新的出題 LLM 模型")
    answer_user_prompt_text: Optional[str] = Field(None, description="新的批改 user prompt")
    answer_llm_model: Optional[str] = Field(None, description="新的批改 LLM 模型")


class BankGroupForExamRequest(BaseModel):
    """PUT /bank/groups/{bank_group_id}/for-exam：設定 for_exam 旗標。"""

    for_exam: bool = Field(True, description="true＝測驗用、false＝取消")


class BankQaAnswerRequest(BaseModel):
    """POST /bank/qa/{bank_qa_id}/llm-answer：對某一題（Bank_QA，由路徑指定）作答並非同步批改。"""

    answer_content: str = Field("", description="學生作答內容；寫入 Bank_QA.answer_content")


class BankGroupQuestionSystemPromptTextResponse(BaseModel):
    """GET/PUT /bank/groups/{bank_group_id}/question-system-prompt-text 回應。"""

    bank_group_id: int
    question_system_prompt_text: str = ""


class PutBankGroupQuestionSystemPromptTextRequest(BaseModel):
    question_system_prompt_text: str = Field(..., description="Bank_Group.question_system_prompt_text")


class BankGroupQuestionUserPromptTextResponse(BaseModel):
    """GET/PUT /bank/groups/{bank_group_id}/question-user-prompt-text 回應。"""

    bank_group_id: int
    question_user_prompt_text: str = ""


class PutBankGroupQuestionUserPromptTextRequest(BaseModel):
    question_user_prompt_text: str = Field(..., description="Bank_Group.question_user_prompt_text")


class BankGroupAnswerUserPromptTextResponse(BaseModel):
    """GET/PUT /bank/groups/{bank_group_id}/answer-user-prompt-text 回應。"""

    bank_group_id: int
    answer_user_prompt_text: str = ""


class PutBankGroupAnswerUserPromptTextRequest(BaseModel):
    answer_user_prompt_text: str = Field(..., description="Bank_Group.answer_user_prompt_text")
