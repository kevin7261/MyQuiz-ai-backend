import os
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2 # 如果要連 PostgreSQL
# import google.generativeai as genai # 如果用 Gemini

app = FastAPI()

# 模擬從環境變數讀取資料庫連線與 API Key
DB_URL = os.environ.get("DATABASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")

class Query(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "Server is running"}

@app.post("/chat")
async def chat_with_llm(query: Query):
    # 這裡實作呼叫 LLM API 的邏輯
    # 例如：model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content(query.prompt)
    
    return {
        "reply": f"這是來自 API 的回應，你輸入了：{query.prompt}",
        "db_status": "已偵測到資料庫連線" if DB_URL else "未連線資料庫"
    }