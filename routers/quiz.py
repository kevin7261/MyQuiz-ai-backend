"""
Quiz 相關 API。原 GET /quiz/quiz-answers 已改為 GET /test/tests，格式同 GET /rag/rags。
"""

from fastapi import APIRouter

router = APIRouter(prefix="/quiz", tags=["quiz"])
