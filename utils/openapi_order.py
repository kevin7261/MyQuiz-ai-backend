"""
API 路徑在 OpenAPI（Swagger）與文件中的顯示順序。

全部 API 掛在 /v1 之下。層級：認證 → RAG（分頁 → 單元 → 題目 → 課程設定）→ Exam → 弱點分析 → 帳號／其他。
同一路徑多方法時：GET → POST → PUT → PATCH → DELETE。
"""

from __future__ import annotations

# 同一路徑多方法時：GET → POST → PUT → PATCH → DELETE
_METHOD_RANK = {"get": 0, "post": 1, "put": 2, "patch": 3, "delete": 4, "head": 5, "options": 6}

# 路徑由前到後；未列者排在同 prefix 群組之後（字母序）
_API_PATH_ORDER: tuple[str, ...] = (
    # --- 認證 ---
    "/v1/auth/login",
    "/v1/auth/refresh",
    # --- RAG：分頁 ---
    "/v1/rag/pages",
    "/v1/rag/pages/upload-zip",
    "/v1/rag/pages/{rag_page_id}",
    "/v1/rag/pages/{rag_page_id}/units",
    "/v1/rag/pages/{rag_page_id}/build-zip",
    "/v1/rag/pages/{rag_page_id}/build-zip-stream",
    # --- RAG：單元（建置前預覽 → 已建置單元）---
    "/v1/rag/pages/{rag_page_id}/unit-preview/text",
    "/v1/rag/pages/{rag_page_id}/unit-preview/mp3-file",
    "/v1/rag/pages/{rag_page_id}/unit-preview/youtube-url",
    "/v1/rag/units/{rag_unit_id}/text",
    "/v1/rag/units/{rag_unit_id}/mp3-file",
    "/v1/rag/units/{rag_unit_id}/youtube-url",
    # --- RAG：題目 CRUD / 標記 ---
    "/v1/rag/quizzes",
    "/v1/rag/quizzes/{rag_quiz_id}",
    "/v1/rag/quizzes/{rag_quiz_id}/followup",
    "/v1/rag/quizzes/{rag_quiz_id}/for-exam",
    # --- RAG：題目 LLM ---
    "/v1/rag/quizzes/llm-generate",
    "/v1/rag/quizzes/llm-generate-db",
    "/v1/rag/quizzes/llm-generate-followup",
    "/v1/rag/quizzes/llm-generate-followup-db",
    "/v1/rag/quizzes/llm-answer",
    "/v1/rag/quizzes/llm-answer-db",
    "/v1/rag/quizzes/answer-result/{job_id}",
    # --- RAG：課程設定 ---
    "/v1/rag/course-members",
    "/v1/rag/course-members/batch",
    "/v1/rag/course-members/{member_person_id}",
    "/v1/rag/llm-api-key",
    "/v1/rag/llm-api-key/exists",
    "/v1/rag/llm-model",
    "/v1/rag/person-analysis-user-prompt-text",
    "/v1/rag/course-analysis-user-prompt-text",
    # --- Bank（測試題庫，複製自 RAG；僅檔案／單元管理）：分頁 ---
    "/v1/bank/pages",
    "/v1/bank/pages/upload-zip",
    "/v1/bank/pages/{bank_page_id}",
    "/v1/bank/pages/{bank_page_id}/units",
    "/v1/bank/pages/{bank_page_id}/build-zip",
    "/v1/bank/pages/{bank_page_id}/build-zip-stream",
    # --- Bank：單元（建置前預覽 → 已建置單元）---
    "/v1/bank/pages/{bank_page_id}/unit-preview/text",
    "/v1/bank/pages/{bank_page_id}/unit-preview/mp3-file",
    "/v1/bank/pages/{bank_page_id}/unit-preview/youtube-url",
    "/v1/bank/units/{bank_unit_id}/text",
    "/v1/bank/units/{bank_unit_id}/mp3-file",
    "/v1/bank/units/{bank_unit_id}/youtube-url",
    # --- Bank：題組（建立／列表巢狀於 unit；單項以主鍵淺路徑）---
    "/v1/bank/pages/{bank_page_id}/units/{bank_unit_id}/groups",
    "/v1/bank/groups/{bank_group_id}",
    "/v1/bank/groups/{bank_group_id}/question-system-prompt-text",
    "/v1/bank/groups/{bank_group_id}/question-user-prompt-text",
    "/v1/bank/groups/{bank_group_id}/answer-user-prompt-text",
    "/v1/bank/groups/{bank_group_id}/for-exam",
    # --- Bank：題目 LLM（出題／批改）---
    "/v1/bank/groups/{bank_group_id}/qa/llm-generate",
    "/v1/bank/qa/{bank_qa_id}/llm-regenerate",
    "/v1/bank/qa/{bank_qa_id}/llm-answer",
    "/v1/bank/qa/answer-result/{job_id}",
    "/v1/bank/qa/{bank_qa_id}",
    # --- Bank：LLM 設定（bank 專屬，與 rag 分開）---
    "/v1/bank/llm-api-key",
    "/v1/bank/llm-api-key/exists",
    "/v1/bank/llm-model",
    "/v1/bank/question-system-prompt-text",
    "/v1/bank/question-user-prompt-text",
    "/v1/bank/answer-user-prompt-text",
    # --- Quiz（試卷／Test，搭配 bank 出題）：試卷 ---
    "/v1/quiz/pages",
    "/v1/quiz/pages/{quiz_page_id}",
    "/v1/quiz/bank-groups",
    # --- Quiz：題組（建立巢狀於試卷；單項以主鍵淺路徑）---
    "/v1/quiz/pages/{quiz_page_id}/groups",
    "/v1/quiz/groups/{quiz_group_id}",
    "/v1/quiz/groups/{quiz_group_id}/question-system-prompt-text",
    "/v1/quiz/groups/{quiz_group_id}/question-user-prompt-text",
    "/v1/quiz/groups/{quiz_group_id}/answer-user-prompt-text",
    # --- Quiz：題目 LLM（出題／批改）與評分 ---
    "/v1/quiz/groups/{quiz_group_id}/qa/llm-generate",
    "/v1/quiz/qa/{quiz_qa_id}/llm-regenerate",
    "/v1/quiz/qa/{quiz_qa_id}/llm-answer",
    "/v1/quiz/qa/answer-result/{job_id}",
    "/v1/quiz/qa/{quiz_qa_id}/question-rate",
    "/v1/quiz/qa/{quiz_qa_id}/answer-rate",
    "/v1/quiz/qa/{quiz_qa_id}",
    # --- Quiz：追問（對題組對應之 Bank 課程內容發問）---
    "/v1/quiz/groups/{quiz_group_id}/llm-ask",
    "/v1/quiz/groups/{quiz_group_id}/asks",
    "/v1/quiz/asks/{quiz_ask_id}/answer-rate",
    "/v1/quiz/asks/{quiz_ask_id}",
    # --- Quiz：LLM 設定（quiz 專屬，與 bank/exam/rag 分開）---
    "/v1/quiz/llm-api-key",
    "/v1/quiz/llm-api-key/exists",
    "/v1/quiz/llm-model",
    "/v1/quiz/question-system-prompt-text",
    "/v1/quiz/question-user-prompt-text",
    "/v1/quiz/answer-user-prompt-text",
    # --- Exam：分頁 ---
    "/v1/exam/pages",
    "/v1/exam/rag-for-exams",
    "/v1/exam/pages/{exam_page_id}",
    # --- Exam：題目 CRUD ---
    "/v1/exam/quizzes/{exam_quiz_id}",
    "/v1/exam/quizzes/{exam_quiz_id}/quiz-rate",
    "/v1/exam/quizzes/{exam_quiz_id}/answer-rate",
    # --- Exam：題目 LLM ---
    "/v1/exam/quizzes/llm-generate",
    "/v1/exam/quizzes/llm-generate-followup",
    "/v1/exam/quizzes/create-llm-generate",
    "/v1/exam/quizzes/create-llm-generate-followup",
    "/v1/exam/quizzes/llm-answer",
    "/v1/exam/quizzes/answer-result/{job_id}",
    # --- Exam：課程設定 ---
    "/v1/exam/llm-api-key",
    "/v1/exam/llm-api-key/exists",
    # --- 弱點分析 ---
    "/v1/person-analyses",
    "/v1/person-analyses/{person_analysis_id}",
    "/v1/person-analyses/{person_analysis_id}/llm-analysis",
    "/v1/course-analyses",
    "/v1/course-analyses/{course_analysis_id}",
    "/v1/course-analyses/{course_analysis_id}/llm-analysis",
    # --- 帳號／其他 ---
    "/v1/users",
    "/v1/users/me/password",
    "/v1/colleges",
    "/v1/courses",
    "/v1/prompt-templates",
    "/v1/logs",
)

_PATH_RANK = {p: i for i, p in enumerate(_API_PATH_ORDER)}


def _path_group_rank(path: str) -> tuple:
    """未在表內的路徑仍依 /v1/rag、/v1/exam 等群組聚在一起。"""
    if path in _PATH_RANK:
        return (0, _PATH_RANK[path], path)
    if path.startswith("/v1/rag/"):
        return (1, 0, path)
    if path.startswith("/v1/quiz/"):
        return (1, 1, path)
    if path.startswith("/v1/exam/"):
        return (1, 2, path)
    if path.startswith("/v1/person-analyses") or path.startswith("/v1/course-analyses"):
        return (1, 3, path)
    return (2, 0, path)


def sort_openapi_paths(paths: dict) -> dict:
    """回傳路徑已排序的 OpenAPI paths 物件（各 path 內 methods 亦排序）。"""
    sorted_path_keys = sorted(paths.keys(), key=_path_group_rank)
    out: dict = {}
    for path in sorted_path_keys:
        item = paths[path]
        methods = sorted(
            item.keys(),
            key=lambda m: (_METHOD_RANK.get(m.lower(), 99), m),
        )
        out[path] = {m: item[m] for m in methods}
    return out
