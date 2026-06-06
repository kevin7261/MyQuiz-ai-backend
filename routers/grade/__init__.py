"""
RAG 出題、評分、題目標記與課程設定 API（路徑順序見 utils.openapi_order）。

**題目標記**（在 llm-generate 之前，與 Exam 題目 CRUD 區塊對齊）：
- PUT /v1/rag/quizzes/{rag_quiz_id}/followup：更新 follow_up
- PUT /v1/rag/quizzes/{rag_quiz_id}/for-exam：更新 for_exam

**出題**：POST …/llm-generate(-db) → POST …/llm-generate-followup(-db)

**評分**：POST …/llm-grade(-db) → GET …/grade-result/{job_id}

**設定**：GET/PUT /v1/rag/llm-api-key、/v1/rag/llm-model；個人／課程分析 prompt 見 course_settings 路由。
"""

from .routes import router

__all__ = ["router"]
