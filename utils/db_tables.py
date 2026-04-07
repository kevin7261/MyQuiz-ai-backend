"""
與 Supabase／Postgres 實際 schema 一致的表名常數。
若遷移或重命名表，僅需改此處。
"""

# public."User"：DDL 為 CREATE TABLE public."User" (...)；PostgREST 使用字串 "User" 對應該表。
USER_TABLE = "User"
