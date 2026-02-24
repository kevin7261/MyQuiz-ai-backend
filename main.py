import os

# 避免 FAISS/NumPy 等載入時出現多份 OpenMP runtime 的 OMP Error #15（macOS 常見）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.zip import router as zip_router
from routers.grade import router as grade_router

app = FastAPI()

# 允許前端跨域呼叫 API，避免瀏覽器 CORS / "Failed to fetch"
# 註：502 時回應可能來自 Render 代理，沒有 CORS 標頭；服務喚醒後正常回應會帶上
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://kevin7261.github.io",  # GitHub Pages 前端
        # 若前端是專案站如 https://kevin7261.github.io/aiquiz_frontend 再補上
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(zip_router)
app.include_router(grade_router)


@app.get("/")
def read_root():
    return {"status": "Server is running"}

