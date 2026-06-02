"""
RAG / Exam API 路徑在 OpenAPI（Swagger）與文件中的顯示順序。

層級：分頁（tab）→ 單元（unit，僅 RAG）→ 題目（quiz）→ 設定（api_key 等）。
RAG 與 Exam 的 quiz 區塊順序對齊：create → name → delete → 標記 → 出題 → 評分 → 其他。
"""

from __future__ import annotations

# 同一路徑多方法時：GET → POST → PUT → PATCH → DELETE
_METHOD_RANK = {"get": 0, "post": 1, "put": 2, "patch": 3, "delete": 4, "head": 5, "options": 6}

# 路徑由前到後；未列者排在同 prefix 群組之後（字母序）
_API_PATH_ORDER: tuple[str, ...] = (
    # --- RAG：分頁 ---
    "/rag/tabs",
    "/rag/tab/units",
    "/rag/tab/create",
    "/rag/tab/create-upload-zip",
    "/rag/tab/tab-name",
    "/rag/tab/delete/{rag_tab_id}",
    "/rag/tab/upload-zip",
    "/rag/tab/build-rag-zip",
    "/rag/tab/build-rag-zip-stream",
    # --- RAG：單元 ---
    "/rag/tab/unit/unit-name",
    "/rag/tab/unit/mp3-file",
    "/rag/tab/unit/youtube-url",
    # --- RAG：題目 CRUD / 標記 ---
    "/rag/tab/unit/quiz/create",
    "/rag/tab/unit/quiz/quiz-name",
    "/rag/tab/unit/quiz/delete/{rag_quiz_id}",
    "/rag/tab/unit/quiz/followup",
    "/rag/tab/unit/quiz/for-exam",
    # --- RAG：題目 LLM ---
    "/rag/tab/unit/quiz/llm-generate",
    "/rag/tab/unit/quiz/llm-generate-db",
    "/rag/tab/unit/quiz/llm-generate-followup",
    "/rag/tab/unit/quiz/llm-generate-followup-db",
    "/rag/tab/unit/quiz/llm-grade",
    "/rag/tab/unit/quiz/llm-grade-db",
    "/rag/tab/unit/quiz/grade-result/{job_id}",
    # --- RAG：單元資源（舊路徑）---
    "/rag/unit/text",
    "/rag/unit/mp3-file",
    "/rag/unit/youtube-url",
    # --- RAG：課程設定 ---
    "/rag/api_key",
    "/rag/person_analysis_user_prompt_text",
    "/rag/course_analysis_user_prompt_text",
    # --- Exam：分頁 ---
    "/exam/tabs",
    "/exam/rag-for-exams",
    "/exam/tab/create",
    "/exam/tab/tab-name",
    "/exam/tab/delete/{exam_tab_id}",
    # --- Exam：題目 CRUD ---
    "/exam/tab/quiz/create",
    "/exam/tab/quiz/delete/{exam_quiz_id}",
    # --- Exam：題目 LLM ---
    "/exam/tab/quiz/llm-generate",
    "/exam/tab/quiz/llm-generate-followup",
    "/exam/tab/quiz/create-llm-generate",
    "/exam/tab/quiz/create-llm-generate-followup",
    "/exam/tab/quiz/llm-grade",
    "/exam/tab/quiz/grade",
    "/exam/tab/quiz/grade-result/{job_id}",
    "/exam/tab/quiz/rate",
    # --- Exam：課程設定 ---
    "/exam/api_key",
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
