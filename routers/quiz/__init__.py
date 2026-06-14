"""Quiz（測驗／Test）API 模組。對應 public.Quiz／Quiz_Group／Quiz_QA；定位等同 exam 之於 rag，但搭配 bank。

Quiz（測驗）→ Quiz_Group（自既有 Bank_Group 快照之題組）→ Quiz_QA（逐題出題／批改，無追問）。
出題／批改沿用 bank 的 LLM 管線（services.bank_generation／services.bank_answering）與 bank 的內容
（RAG ZIP／逐字稿）；金鑰／模型走 quiz- 設定（/v1/quiz/llm-api-key、/v1/quiz/llm-model）。程式不與 exam／rag 共用。

兩個子路由器（皆掛 /quiz）：
1. 測驗／題組／出題／批改／評分（.routes）：
   測驗：GET/POST /quiz/pages → PATCH/DELETE /quiz/pages/{quiz_page_id}
   可選 Bank 題組：GET /quiz/bank-groups（for_exam=true）
   題組：POST /quiz/pages/{quiz_page_id}/groups（挑 bank_group_id 快照）；GET/PATCH/DELETE /quiz/groups/{quiz_group_id}
   出題：POST /quiz/groups/{quiz_group_id}/qa/llm-generate（逐題，上限 qa_count）；POST /quiz/qa/{quiz_qa_id}/llm-regenerate
   批改：POST /quiz/qa/{quiz_qa_id}/llm-answer → GET /quiz/qa/answer-result/{job_id}
   評分／刪題：PUT /quiz/qa/{quiz_qa_id}/{question-rate,answer-rate}；DELETE /quiz/qa/{quiz_qa_id}
   追問（對題組對應之 Bank 課程內容發問，寫入 Quiz_Ask）：POST /quiz/groups/{quiz_group_id}/llm-ask
   → GET /quiz/groups/{quiz_group_id}/asks；PUT /quiz/asks/{quiz_ask_id}/answer-rate；DELETE /quiz/asks/{quiz_ask_id}
2. LLM 設定（.settings_routes）：GET/PUT /quiz/llm-api-key、GET /quiz/llm-api-key/exists、GET/PUT /quiz/llm-model。
"""

from fastapi import APIRouter

from .routes import router as _routes_router
from .settings_routes import router as _settings_router

router = APIRouter()
router.include_router(_routes_router)
router.include_router(_settings_router)

__all__ = ["router"]
