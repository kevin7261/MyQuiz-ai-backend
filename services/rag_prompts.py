"""
RAG 向量檢索設定（POST /v1/rag/pages/{rag_page_id}/build-zip 建 FAISS 後，llm-generate／llm-grade unit_type=1 使用）。
建庫本身僅 embedding／切分，無 Chat LLM prompt；此模組僅描述 retriever 查詢句與相關常數。
"""

from services.grading import (
    GRADE_EMBEDDING_MODEL,
    GRADE_RAG_CHUNK_OVERLAP,
    GRADE_RAG_CHUNK_SIZE,
    GRADE_RETRIEVAL_K,
)
from services.quiz_generation import DEFAULT_RETRIEVAL_QUERY, EMBEDDING_MODEL, RETRIEVAL_K

LLM_GRADE_RAG_RETRIEVAL_QUERY_TEMPLATE = "{quiz_content}"


def rag_retrieval_config() -> dict[str, dict[str, int | str]]:
    """回傳 unit_type=1 向量檢索查詢句與 retrieval_k。"""
    return {
        "llm_generate": {
            "retrieval_query": DEFAULT_RETRIEVAL_QUERY,
            "retrieval_k": RETRIEVAL_K,
        },
        "llm_grade": {
            "retrieval_query": LLM_GRADE_RAG_RETRIEVAL_QUERY_TEMPLATE,
            "retrieval_k": GRADE_RETRIEVAL_K,
        },
    }


def rag_build_defaults() -> dict[str, int | str]:
    """POST /v1/rag/pages/{rag_page_id}/build-zip 建 FAISS 預設參數（供 prompt API 對照）。"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size_default": GRADE_RAG_CHUNK_SIZE,
        "chunk_overlap_default": GRADE_RAG_CHUNK_OVERLAP,
        "grade_embedding_model": GRADE_EMBEDDING_MODEL,
    }
