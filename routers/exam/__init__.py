"""
Exam API 模組。對應 public.Exam、public.Exam_Quiz。路徑層級與 RAG 對齊（見 utils.openapi_order）。

檔案結構：模型／檢索說明 → LLM Prompt → 型別與作業快取 → Pydantic → 輔助函式與路由。

**分頁**：GET /exam/pages → GET /exam/rag-for-exams → POST /exam/pages → PATCH /exam/pages/{exam_page_id}
→ DELETE /exam/pages/{exam_page_id}

**題目**：POST /exam/quizzes → DELETE /exam/quizzes/{exam_quiz_id}
→ POST /exam/quizzes/llm-generate → POST /exam/quizzes/llm-generate-followup
→ POST /exam/quizzes/create-llm-generate → POST /exam/quizzes/create-llm-generate-followup
→ POST /exam/quizzes/llm-grade → GET /exam/quizzes/grade-result/{job_id} → PUT /exam/quizzes/{exam_quiz_id}/quiz-rate → PUT /exam/quizzes/{exam_quiz_id}/grade-rate

**設定**：GET/PUT /exam/llm-api-key

Exam_Quiz 僅回傳未軟刪（deleted=true 不回傳）。刪除題目會一併軟刪追問子題鏈。
"""

from .routes import router

__all__ = ["router"]
