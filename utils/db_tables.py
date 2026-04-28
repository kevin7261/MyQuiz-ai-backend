"""
與 Supabase／Postgres 實際 schema 一致的表名常數。
若遷移或重命名表，僅需改此處。
"""

# public."User"
USER_TABLE = "User"

# RAG 相關表名
RAG_TABLE = "Rag"
RAG_UNIT_TABLE = "Rag_Unit"
RAG_QUIZ_TABLE = "Rag_Quiz"
