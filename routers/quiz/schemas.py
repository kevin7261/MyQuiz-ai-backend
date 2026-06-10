"""routers.quiz（試卷／Test）schemas。

Quiz（試卷）→ Quiz_Group（自既有 Bank_Group 快照之題組）→ Quiz_QA（逐題出題／批改，無追問）。
與 bank 搭配：出題／批改沿用 bank 的 LLM 管線與內容（RAG ZIP／逐字稿），金鑰／模型走 quiz- 設定。
程式不與 exam／rag 共用。
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


QuizRateValue = Literal[-1, 0, 1]


# ---------------------------------------------------------------------------
# 試卷（Quiz）
# ---------------------------------------------------------------------------


class CreateQuizRequest(BaseModel):
    """POST /quiz/pages：建立一份試卷（Quiz）。"""

    quiz_page_id: str | None = Field(None, description="選填；未傳則由後端產生")
    person_id: str = Field("", description="選填；未傳以 token 解析的呼叫者為準；有傳須與呼叫者一致")
    tab_name: str = Field("", description="試卷顯示名稱")


class UpdateQuizTabNameRequest(BaseModel):
    """PATCH /quiz/pages/{quiz_page_id}：更新 tab_name。"""

    tab_name: str = Field(..., description="新的顯示名稱")


class ListQuizResponse(BaseModel):
    """GET /quiz/pages 回應：每筆 Quiz 含 quiz_groups[]（每組含 qas[]）。"""

    quizzes: list[dict] = Field(
        ...,
        description="每筆 Quiz 的 quiz_groups[] 為 Quiz_Group（含 qas[]＝Quiz_QA，依 question_series_index 升序）",
    )
    count: int


# ---------------------------------------------------------------------------
# 題組（Quiz_Group，自 Bank_Group 快照）
# ---------------------------------------------------------------------------


class CreateQuizGroupRequest(BaseModel):
    """POST /quiz/pages/{quiz_page_id}/groups：挑選一個既有 Bank_Group，快照成本試卷下的 Quiz_Group。"""

    bank_group_id: int = Field(..., gt=0, description="來源 Bank_Group 主鍵（>0）；其設定將快照進新 Quiz_Group")
    group_name: str = Field("", description="選填；題組顯示名稱（空則沿用 Bank_Group.group_name）")


class UpdateQuizGroupRequest(BaseModel):
    """PATCH /quiz/groups/{quiz_group_id}：更新題組快照（僅更新有傳入的欄位）。"""

    group_name: Optional[str] = Field(None, description="新的題組顯示名稱")
    qa_count: Optional[int] = Field(None, ge=0, description="新的題數上限")
    question_system_prompt_text: Optional[str] = Field(None, description="新的連續出題規定")
    question_user_prompt_text: Optional[str] = Field(None, description="新的出題 user prompt")
    question_llm_model: Optional[str] = Field(None, description="新的出題 LLM 模型")
    answer_user_prompt_text: Optional[str] = Field(None, description="新的批改 user prompt")
    answer_llm_model: Optional[str] = Field(None, description="新的批改 LLM 模型")


class ListQuizBankGroupsResponse(BaseModel):
    """GET /quiz/bank-groups 回應：可選用的 Bank_Group（for_exam=true）列，附單元資訊。"""

    groups: list[dict] = Field(
        ...,
        description="Bank_Group 列（含 bank_group_id、bank_page_id、bank_unit_id、unit_name、unit_type、group_name、qa_count）",
    )
    count: int


# ---------------------------------------------------------------------------
# 逐題出題（Quiz_QA；無追問）
# ---------------------------------------------------------------------------


class GenerateQuizQaRequest(BaseModel):
    """
    POST /quiz/groups/{quiz_group_id}/qa/llm-generate：在題組內產生下一題（LLM）。
    一律使用該題組既有之 question_system_prompt_text／question_user_prompt_text；
    本次可選擇性覆寫（非空才覆寫，不寫回 DB）。
    """

    question_user_prompt_text: str = Field(
        "", description="可選；本次出題覆寫用的 user prompt（空則用題組既有值）"
    )
    question_system_prompt_text: str = Field(
        "", description="可選；本次連續出題規定覆寫（空則用題組既有值）"
    )


# ---------------------------------------------------------------------------
# 批改（Quiz_QA；非同步）與評分
# ---------------------------------------------------------------------------


class QuizQaAnswerRequest(BaseModel):
    """POST /quiz/qa/{quiz_qa_id}/llm-answer：對某一題（Quiz_QA，由路徑指定）作答並非同步批改。"""

    answer_content: str = Field("", description="學生作答內容；寫入 Quiz_QA.answer_content")


class QuizQaQuestionRateRequest(BaseModel):
    """PUT /quiz/qa/{quiz_qa_id}/question-rate：更新 Quiz_QA.question_rate。"""

    question_rate: QuizRateValue = Field(0, description="僅 -1、0、1")


class QuizQaAnswerRateRequest(BaseModel):
    """PUT /quiz/qa/{quiz_qa_id}/answer-rate：更新 Quiz_QA.answer_rate。"""

    answer_rate: QuizRateValue = Field(0, description="僅 -1、0、1")
