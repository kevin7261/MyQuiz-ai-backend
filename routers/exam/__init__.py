"""
Exam API 模組。對應 public.Exam、public.Exam_Quiz。路徑層級與 RAG 對齊（見 utils.openapi_order）。

檔案結構：模型／檢索說明 → LLM Prompt → 型別與作業快取 → Pydantic → 輔助函式與路由。

**分頁**：GET /exam/pages → GET /exam/rag-for-exams → POST /exam/page/add → PUT /exam/page/tab-name
→ PUT /exam/page/delete/{exam_page_id}

**題目**：POST /exam/page/quiz/add → PUT /exam/page/quiz/delete/{exam_quiz_id}
→ POST /exam/page/quiz/llm-generate → POST /exam/page/quiz/llm-generate-followup
→ POST /exam/page/quiz/create-llm-generate → POST /exam/page/quiz/create-llm-generate-followup
→ POST /exam/page/quiz/llm-grade → GET /exam/page/quiz/grade-result/{job_id} → POST /exam/page/quiz/quiz-rate → POST /exam/page/quiz/grade-rate

**設定**：GET/PUT /exam/llm_api_key

Exam_Quiz 僅回傳未軟刪（deleted=true 不回傳）。刪除題目會一併軟刪追問子題鏈。
"""

from .routes import router

__all__ = ["router"]
