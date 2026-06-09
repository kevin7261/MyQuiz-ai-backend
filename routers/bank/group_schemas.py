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


class CreateBankGroupRequest(BaseModel):
    """
    POST /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups：新增一個測試題組（Bank_Group），**不呼叫 LLM**。
    所屬單元由路徑 bank_page_id／bank_unit_id 決定。
    """

    group_name: str = Field("", description="題組顯示名稱")
    qa_count: int = Field(
        0, ge=0, description="本題組預定要出的題數上限；逐題產生時不會超過此數（0 表示不限）"
    )
    question_system_prompt_text: str = Field(
        "", description="連續出題的規定（織入出題 system prompt，最高優先；如越來越難、勿重複）"
    )
    question_user_prompt_text: str = Field("", description="出題規定（出題 user prompt）")
    question_llm_model: str = Field("", description="出題 LLM 模型；空則用課程 llm-model 設定")
    answer_user_prompt_text: str = Field("", description="批改規定（批改 user prompt）")
    answer_llm_model: str = Field("", description="批改 LLM 模型；空則用課程 llm-model 設定")
    for_exam: bool = Field(False, description="是否標記為測驗用")


class UpdateBankGroupRequest(BaseModel):
    """PATCH /bank/groups/{bank_group_id}：更新題組設定（僅更新有傳入的欄位）。"""

    group_name: Optional[str] = Field(None, description="新的題組顯示名稱")
    qa_count: Optional[int] = Field(None, ge=0, description="新的題數上限")
    question_system_prompt_text: Optional[str] = Field(None, description="新的連續出題規定")
    question_user_prompt_text: Optional[str] = Field(None, description="新的出題 user prompt")
    question_llm_model: Optional[str] = Field(None, description="新的出題 LLM 模型")
    answer_user_prompt_text: Optional[str] = Field(None, description="新的批改 user prompt")
    answer_llm_model: Optional[str] = Field(None, description="新的批改 LLM 模型")


class BankGroupForExamRequest(BaseModel):
    """PUT /bank/groups/{bank_group_id}/for-exam：設定 for_exam 旗標。"""

    for_exam: bool = Field(True, description="true＝測驗用、false＝取消")


class GenerateBankQaRequest(BaseModel):
    """
    POST /bank/groups/{bank_group_id}/qa/llm-generate：在題組內產生下一題（LLM）。
    一律使用該題組既有之 question_system_prompt_text／question_user_prompt_text；
    本次可選擇性覆寫（非空才覆寫，不寫回 DB）。
    """

    question_user_prompt_text: str = Field(
        "", description="可選；本次出題覆寫用的 user prompt（空則用題組既有值）"
    )
    question_system_prompt_text: str = Field(
        "", description="可選；本次連續出題規定覆寫（空則用題組既有值）"
    )


class BankQaAnswerRequest(BaseModel):
    """POST /bank/qa/{bank_qa_id}/llm-answer：對某一題（Bank_QA，由路徑指定）作答並非同步批改。"""

    answer_content: str = Field("", description="學生作答內容；寫入 Bank_QA.answer_content")
