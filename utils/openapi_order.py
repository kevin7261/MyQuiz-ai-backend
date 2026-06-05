"""
RAG / Exam API 路徑在 OpenAPI（Swagger）與文件中的顯示順序。

層級：分頁（page）→ 單元（unit，僅 RAG）→ 題目（quiz）→ 設定（api_key 等）。
RAG 與 Exam 的 quiz 區塊順序對齊：add → name → delete → 標記 → 出題 → 評分 → 其他。
"""

from __future__ import annotations

# 同一路徑多方法時：GET → POST → PUT → PATCH → DELETE
_METHOD_RANK = {"get": 0, "post": 1, "put": 2, "patch": 3, "delete": 4, "head": 5, "options": 6}

# 路徑由前到後；未列者排在同 prefix 群組之後（字母序）
_API_PATH_ORDER: tuple[str, ...] = (
    # --- RAG：分頁 ---
    "/rag/pages",
    "/rag/page/units",
    "/rag/page/add-upload-zip",
    "/rag/page/tab-name",
    "/rag/page/delete/{rag_page_id}",
    "/rag/page/build-rag-zip",
    "/rag/page/build-rag-zip-stream",
    # --- RAG：單元 ---
    "/rag/page/unit/mp3-file",
    # --- RAG：題目 CRUD / 標記 ---
    "/rag/page/unit/quiz/add",
    "/rag/page/unit/quiz/quiz-name",
    "/rag/page/unit/quiz/delete/{rag_quiz_id}",
    "/rag/page/unit/quiz/followup",
    "/rag/page/unit/quiz/for-exam",
    # --- RAG：題目 LLM ---
    "/rag/page/unit/quiz/llm-generate",
    "/rag/page/unit/quiz/llm-generate-db",
    "/rag/page/unit/quiz/llm-generate-followup",
    "/rag/page/unit/quiz/llm-generate-followup-db",
    "/rag/page/unit/quiz/llm-grade",
    "/rag/page/unit/quiz/llm-grade-db",
    "/rag/page/unit/quiz/grade-result/{job_id}",
    # --- RAG：單元資源（舊路徑）---
    "/rag/unit/text",
    "/rag/unit/mp3-file",
    "/rag/unit/youtube-url",
    # --- RAG：課程設定 ---
    "/rag/course-members",
    "/rag/course-members/add",
    "/rag/course-members/edit/{person_id}",
    "/rag/course-members/delete/{person_id}",
    "/rag/llm_api_key",
    "/rag/llm_model",
    "/rag/person_analysis_user_prompt_text",
    "/rag/course_analysis_user_prompt_text",
    # --- Exam：分頁 ---
    "/exam/pages",
    "/exam/rag-for-exams",
    "/exam/page/add",
    "/exam/page/tab-name",
    "/exam/page/delete/{exam_page_id}",
    # --- Exam：題目 CRUD ---
    "/exam/page/quiz/delete/{exam_quiz_id}",
    # --- Exam：題目 LLM ---
    "/exam/page/quiz/llm-generate",
    "/exam/page/quiz/llm-generate-followup",
    "/exam/page/quiz/create-llm-generate",
    "/exam/page/quiz/create-llm-generate-followup",
    "/exam/page/quiz/llm-grade",
    "/exam/page/quiz/grade",
    "/exam/page/quiz/grade-result/{job_id}",
    "/exam/page/quiz/quiz-rate",
    "/exam/page/quiz/grade-rate",
    # --- Exam：課程設定 ---
    "/exam/llm_api_key",
    # --- 弱點分析 ---
    "/person-analysis/analysis",
    "/person-analysis/llm-analysis",
    "/course-analysis/analysis",
    "/course-analysis/llm-analysis",
    # --- 帳號／個人檔案 ---
    "/profile/users",
    "/profile/users/batch",
    "/profile/users/delete",
    "/profile/password",
    "/profile/login",
)

_PATH_RANK = {p: i for i, p in enumerate(_API_PATH_ORDER)}


def _path_group_rank(path: str) -> tuple:
    """未在表內的路徑仍依 /rag、/exam 等群組聚在一起。"""
    if path in _PATH_RANK:
        return (0, _PATH_RANK[path], path)
    if path.startswith("/rag/"):
        return (1, 0, path)
    if path.startswith("/exam/"):
        return (1, 1, path)
    if path.startswith("/person-analysis/") or path.startswith("/course-analysis/"):
        return (1, 2, path)
    if path.startswith("/profile"):
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
