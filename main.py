"""
MyQuiz.ai 後端入口。

啟動流程：
  1. load_dotenv() 載入 .env（SUPABASE_URL、SUPABASE_SERVICE_ROLE_KEY 等）
  2. 修正 macOS 多 OpenMP runtime 衝突（KMP_DUPLICATE_LIB_OK）
  3. 建立 FastAPI app，掛載 CORS 與 APILogMiddleware
  4. 依序掛載各子路由器
"""

import os

from dotenv import load_dotenv

load_dotenv()

# 避免 FAISS/NumPy 在 macOS 上因多份 OpenMP runtime 觸發 OMP Error #15
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dependencies.person_id import PersonId
from middleware.api_log_middleware import APILogMiddleware
from routers.course_analysis import router as course_analysis_router
from routers.exam import router as exam_router
from routers.grade import router as grade_router
from routers.log import router as log_router
from routers.person_analysis import router as person_analysis_router
from routers.system_settings import router as system_settings_router
from routers.users import router as users_router
from routers.zip import router as zip_router

# ---------------------------------------------------------------------------
# FastAPI 應用程式
# ---------------------------------------------------------------------------

app = FastAPI(title="MyQuiz.ai_backend")

# ---------------------------------------------------------------------------
# CORS 設定
# ---------------------------------------------------------------------------
# 前端若跑在未列舉的 origin（如自訂埠、區網 IP、Vercel 改名後），
# 可於 .env 或部署平台設定 CORS_EXTRA_ORIGINS（逗號分隔）追加，無需修改程式。
#
# 注意：評分已改為非同步，POST .../llm-grade 回傳 202 + job_id，
# 請用對應的 GET .../grade-result/{job_id} 輪詢結果；
# 若出現 502 表示 Render 代理逾時（約 30 秒），回應不帶 CORS 標頭。

_cors_base = [
    "http://localhost:8080",        # Vue CLI／webpack 常見埠
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:5173",        # Vite 預設埠
    "http://127.0.0.1:5173",
    "http://localhost:4173",        # vite preview
    "http://127.0.0.1:4173",
    "http://localhost:3000",        # React／Next.js 常見本機埠
    "http://127.0.0.1:3000",
    "https://kevin7261.github.io",
    "https://myquiz-ai.vercel.app",
]

_extra = (os.environ.get("CORS_EXTRA_ORIGINS") or "").strip()
if _extra:
    _more = [x.strip() for x in _extra.split(",") if x.strip()]
    _cors_allow = list(dict.fromkeys(_cors_base + _more))
else:
    _cors_allow = _cors_base

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 每次業務 API 請求寫入 public.Log（OPTIONS / HEAD、Swagger 路徑不記錄）
app.add_middleware(APILogMiddleware)

# ---------------------------------------------------------------------------
# 路由掛載
# ---------------------------------------------------------------------------
# zip_router 先於 grade_router 掛載，讓 /rag/tab/unit/quiz/create（無 LLM）
# 在 OpenAPI 文件上排在 llm-generate 之前，視覺上較直觀。

app.include_router(zip_router)
app.include_router(grade_router)
app.include_router(exam_router)
app.include_router(person_analysis_router)
app.include_router(course_analysis_router)
app.include_router(users_router)
app.include_router(system_settings_router)
app.include_router(log_router)


# ---------------------------------------------------------------------------
# 健康檢查
# ---------------------------------------------------------------------------

@app.get("/")
def read_root(_person_id: PersonId):
    """根路徑健康檢查；確認伺服器正常運行。"""
    return {"status": "Server is running"}
