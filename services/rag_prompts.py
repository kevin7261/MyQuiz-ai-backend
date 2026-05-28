"""
RAG 向量檢索「抓片段」設定（POST /rag/tab/build-rag-zip 建 FAISS 後，llm-generate／llm-grade unit_type=1 使用）。
建庫本身僅 embedding／切分，無 Chat LLM system／user 模板；下列 user 為送進 retriever 的查詢句。
"""

from services.grading import (
    GRADE_EMBEDDING_MODEL,
    GRADE_RAG_CHUNK_OVERLAP,
    GRADE_RAG_CHUNK_SIZE,
    GRADE_RETRIEVAL_K,
)
from services.quiz_generation import DEFAULT_RETRIEVAL_QUERY, EMBEDDING_MODEL, RETRIEVAL_K

LLM_GRADE_RAG_RETRIEVAL_QUERY_TEMPLATE = "{quiz_content}"


def rag_prompt_templates() -> dict[str, dict[str, str]]:
    """回傳抓 RAG（向量檢索）之 system／user 查詢模板。"""
    return {
        "llm_generate": {
            "system": "",
            "user": DEFAULT_RETRIEVAL_QUERY,
        },
        "llm_grade": {
            "system": "",
            "user": LLM_GRADE_RAG_RETRIEVAL_QUERY_TEMPLATE,
        },
    }


def rag_build_defaults() -> dict[str, int | str]:
    """POST /rag/tab/build-rag-zip 建 FAISS 預設參數（供 prompt API 對照）。"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size_default": GRADE_RAG_CHUNK_SIZE,
        "chunk_overlap_default": GRADE_RAG_CHUNK_OVERLAP,
        "retrieval_k_llm_generate": RETRIEVAL_K,
        "retrieval_k_llm_grade": GRADE_RETRIEVAL_K,
        "grade_embedding_model": GRADE_EMBEDDING_MODEL,
    }
