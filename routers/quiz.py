"""
Quiz 相關 API 模組。
原 GET /quiz/quiz-answers 已改為 GET /exam/exams，格式同 GET /rag/rags。
此路由目前為空殼，保留供未來擴充。
"""

# 引入 FastAPI 的 APIRouter
from fastapi import APIRouter

# 建立路由，前綴為 /quiz，標籤為 quiz
router = APIRouter(prefix="/quiz", tags=["quiz"])
