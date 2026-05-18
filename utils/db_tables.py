"""
與 Supabase／Postgres 實際 schema 一致的表名常數。
若遷移或重命名表，僅需改此處。
"""

# public."User"
USER_TABLE = "User"
# enrollments / per-course profile（user_type、llm_api_key）
USER_COURSE_RELATION_TABLE = "User_Course_Relation"

# Supabase PostgREST：deleted 為 false 或 null 視為有效列（與舊資料相容）
ACTIVE_DELETED_FILTER = "deleted.eq.false,deleted.is.null"

# RAG 相關表名
RAG_TABLE = "Rag"
RAG_UNIT_TABLE = "Rag_Unit"
RAG_QUIZ_TABLE = "Rag_Quiz"
