from fastapi import FastAPI
from pydantic import BaseModel

from routers.zip import router as zip_router

app = FastAPI()

app.include_router(zip_router)


class Query(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"status": "Server is running"}


@app.post("/chat")
async def chat_with_llm(query: Query):
    return {
        "reply": f"這是來自 API 的回應，你輸入了：{query.prompt}"
    }

