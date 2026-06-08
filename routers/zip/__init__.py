"""
ZIP 與 RAG 相關 API 模組。路徑層級與排序與 Exam 對齊（見 utils.openapi_order、README API 目錄）。

**分頁**：GET /rag/pages → GET /rag/pages/{rag_page_id}/units → POST /rag/pages/upload-zip
→ PATCH /rag/pages/{rag_page_id} → DELETE /rag/pages/{rag_page_id} → POST /rag/pages/{rag_page_id}/build-zip（-stream 別名）

**單元**：GET /rag/units/{rag_unit_id}/text → GET /rag/units/{rag_unit_id}/mp3-file → GET /rag/units/{rag_unit_id}/youtube-url

**建置前預覽**（upload 後、build 前，Rag_Unit 尚無列；以 folder_name 讀 upload ZIP）：
GET /rag/pages/{rag_page_id}/unit-preview/text、…/unit-preview/mp3-file、…/unit-preview/youtube-url

**題目**：POST /rag/quizzes → PATCH /rag/quizzes/{rag_quiz_id} → DELETE /rag/quizzes/{rag_quiz_id}
→（followup／for-exam／llm-* 見 routers/answer）

GET /rag/pages 須 course_id、person_id；`local` 未傳時依連線判定。build-rag-zip 見路由 docstring。
"""

from .routes import router

__all__ = ["router"]
