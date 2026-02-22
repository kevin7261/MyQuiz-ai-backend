import os

# 避免 FAISS/NumPy 等載入時出現多份 OpenMP runtime 的 OMP Error #15（macOS 常見）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.zip import router as zip_router

app = FastAPI()

# 允許前端（不同 port）呼叫 API，避免瀏覽器「failed to fetch」
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(zip_router)


@app.get("/")
def read_root():
    return {"status": "Server is running"}

