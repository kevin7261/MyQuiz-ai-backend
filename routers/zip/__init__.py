"""
ZIP 與 RAG 相關 API 模組。路徑層級與排序與 Exam 對齊（見 utils.openapi_order、README API 目錄）。

**分頁**：GET /rag/pages → GET /rag/page/units → POST /rag/page/add-upload-zip
→ PUT /rag/page/tab-name → PUT /rag/page/delete/{rag_page_id} → POST /rag/page/upload-zip → POST /rag/page/build-rag-zip（-stream 別名）

**單元**：PUT /rag/page/unit/unit-name → GET /rag/page/unit/mp3-file → GET /rag/page/unit/youtube-url

**題目**：POST /rag/page/unit/quiz/add → PUT /rag/page/unit/quiz/quiz-name → PUT /rag/page/unit/quiz/delete/{rag_quiz_id}
→（followup／for-exam／llm-* 見 routers/grade）

**舊路徑**：GET /rag/unit/text、/rag/unit/mp3-file、/rag/unit/youtube-url

GET /rag/pages 須 course_id、person_id；`local` 未傳時依連線判定。build-rag-zip 見路由 docstring。
"""

from .routes import router

__all__ = ["router"]
