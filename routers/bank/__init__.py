"""
Bank（測試題庫）API 模組。自 routers.zip（/rag）複製之獨立題庫，程式不與 rag 共用；操作 Bank／Bank_Unit／Bank_Group／Bank_QA 表。

兩個子路由器（皆掛 /bank）：
1. 檔案／單元管理（.routes）— 自 rag 檔案／單元管理複製。
   分頁：GET /bank/pages → GET /bank/pages/{bank_page_id}/units → POST /bank/pages/upload-zip
   → PATCH /bank/pages/{bank_page_id} → DELETE /bank/pages/{bank_page_id} → POST /bank/pages/{bank_page_id}/build-zip
   單元：GET /bank/units/{bank_unit_id}/{text,mp3-file,youtube-url}；建置前預覽 /bank/pages/{bank_page_id}/unit-preview/*

2. 題組／問答（.group_routes）— 對應 rag 的 Rag_Quiz LLM 出題／批改，但以「題組」為單位、無追問。
   階層 bank → page → unit → group → qa；URL 採業界慣例：建立／列表巢狀，單項以主鍵淺路徑（與 rag 一致）。
   建／列題組：POST／GET /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups（設 qa_count 與出題／批改 prompt）。
   單一題組：GET／PATCH／DELETE /bank/groups/{bank_group_id}、PUT /bank/groups/{bank_group_id}/for-exam。
   題組內出題：POST /bank/groups/{bank_group_id}/qa/llm-generate（逐題，上限 qa_count）。
   單題批改：POST /bank/qa/{bank_qa_id}/llm-answer → GET /bank/qa/answer-result/{job_id}；刪題 DELETE /bank/qa/{bank_qa_id}。
   LLM API Key／模型用 bank 專屬課程設定（/v1/bank/llm-api-key、/v1/bank/llm-model；Course_Setting key=bank-api-key／bank-llm-model），與 rag、exam 完全分開。
"""

from fastapi import APIRouter

from .routes import router as _files_router
from .group_routes import router as _group_router
from .settings_routes import router as _settings_router

router = APIRouter()
router.include_router(_files_router)
router.include_router(_group_router)
router.include_router(_settings_router)

__all__ = ["router"]
