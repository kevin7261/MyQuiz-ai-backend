import os

# 從專案根目錄的 .env 載入環境變數（SUPABASE_URL、SUPABASE_SERVICE_ROLE_KEY 等）
from dotenv import load_dotenv
load_dotenv()

# 避免 FAISS/NumPy 等載入時出現多份 OpenMP runtime 的 OMP Error #15（macOS 常見）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.zip import router as zip_router
from routers.grade import router as grade_router
from routers.quiz import router as quiz_router
from routers.exam import router as exam_router
from routers.users import router as users_router

app = FastAPI()

# 允許前端跨域呼叫 API，避免瀏覽器 CORS / "Failed to fetch"
# 註：若出現 502，回應來自 Render 代理（逾時約 30s），不會帶 CORS 標頭；
# 評分已改為非同步：POST /rag/quiz-grade 回傳 202 + job_id，請用 GET /rag/quiz-grade-result/{job_id} 輪詢結果
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://kevin7261.github.io",  # GitHub Pages 前端
        "https://aiquizfrontend.vercel.app",  # Vercel 前端
        # 若前端是專案站如 https://kevin7261.github.io/aiquiz_frontend 再補上
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 順序影響 API 文件顯示：zip_router 先掛載，使 generate-quiz / quiz-grade / quiz-grade-result 出現在 rag 群組最下面
app.include_router(zip_router)
app.include_router(grade_router)
app.include_router(quiz_router)
app.include_router(exam_router)
app.include_router(users_router)


@app.get("/")
def read_root():
    return {"status": "Server is running"}

