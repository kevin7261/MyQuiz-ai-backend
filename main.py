# 引入 Python 內建模組 os，用於存取環境變數與作業系統相關操作
import os

# 從 dotenv 套件載入 load_dotenv 函數，用於從專案根目錄 .env 讀取環境變數
from dotenv import load_dotenv
# 執行 load_dotenv，將 .env 中的變數（如 SUPABASE_URL、SUPABASE_SERVICE_ROLE_KEY）載入 os.environ
load_dotenv()

# 設定 KMP_DUPLICATE_LIB_OK 為 TRUE，避免 FAISS/NumPy 載入時出現多份 OpenMP runtime 的 OMP Error #15（macOS 常見問題）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 引入 FastAPI 框架，用於建立 Web API 服務
from fastapi import FastAPI

# 所有端點必填 query 參數 person_id
from dependencies.person_id import PersonId
# 引入 CORS 中介軟體，用於處理跨域請求
from fastapi.middleware.cors import CORSMiddleware
# 每次請求寫入 public.Log
from middleware.api_log_middleware import APILogMiddleware

# 引入各子模組的路由器，並以別名區分（zip_router 等）
from routers.zip import router as zip_router
# 評分相關路由（含 tab/unit/quiz/create 無 LLM、tab/unit/quiz/llm-generate、tab/unit/quiz/llm-grade、tab/unit/quiz/grade-result）
from routers.grade import router as grade_router
# Exam 測驗相關路由（tabs、rag-for-exams、tab/create、tab/quiz/create、tab/quiz/llm-generate、tab/quiz/llm-grade、grade-result 等）
from routers.exam import router as exam_router
# 個人分析路由（依 person_id 查詢測驗與弱點報告）
from routers.person_analysis import router as person_analysis_router
# 課程分析路由（Exam_Quiz 全部內容）
from routers.course_analysis import router as course_analysis_router
# 使用者相關路由（登入、個人資料、使用者列表）
from routers.users import router as users_router
# 系統設定路由（LLM API Key 的 GET/PUT）
from routers.system_settings import router as system_settings_router
# Log 表查詢路由
from routers.log import router as log_router

# 建立 FastAPI 應用程式實例（OpenAPI /docs 標題）
app = FastAPI(title="MyQuiz.ai_backend")

# 註冊 CORS 中介軟體，允許前端跨域呼叫 API，避免瀏覽器 CORS 或 "Failed to fetch" 錯誤
# 若出現 502，回應來自 Render 代理（逾時約 30 秒），不會帶 CORS 標頭
# 評分已改為非同步：POST /rag/.../llm-grade、POST /exam/.../llm-grade 回傳 202 + job_id，請用對應 GET .../grade-result/{job_id} 輪詢結果
# 後端預設 uvicorn :8000；前端若跑在其它 origin（如 :8081），必須列在下方或 CORS_EXTRA_ORIGINS，否則瀏覽器會擋跨域（與 API 404 無關）
_cors_base = [
    "http://localhost:8080",           # Vue CLI／webpack 常見埠
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:5173",           # Vite 預設
    "http://127.0.0.1:5173",
    "http://localhost:4173",           # vite preview
    "http://127.0.0.1:4173",
    "http://localhost:3000",           # 部分 React／Next 本機
    "http://127.0.0.1:3000",
    "https://kevin7261.github.io",
    "https://myquiz-ai.vercel.app",
]
# Vercel／改專案名後網址會變，可在 .env 或部署平台設定 CORS_EXTRA_ORIGINS（逗號分隔）追加，無需改程式
_extra = (os.environ.get("CORS_EXTRA_ORIGINS") or "").strip()
if _extra:
    _more = [x.strip() for x in _extra.split(",") if x.strip()]
    _cors_allow = list(dict.fromkeys(_cors_base + _more))
else:
    _cors_allow = _cors_base

app.add_middleware(
    # 使用 CORSMiddleware 中介軟體
    CORSMiddleware,
    # 允許的來源網域列表，前端可由此發送請求
    allow_origins=_cors_allow,
    # 允許攜帶 cookie、認證等憑證
    allow_credentials=True,
    # 允許所有 HTTP 方法（GET、POST、PUT、PATCH、DELETE 等）
    allow_methods=["*"],
    # 允許所有 HTTP 標頭
    allow_headers=["*"],
)

app.add_middleware(APILogMiddleware)

# 依序掛載各路由至 app；順序影響 Swagger API 文件顯示（標籤先出現者排在較上）
# zip_router 於 grade 之前掛載，使 tab/unit/quiz/create（無 LLM）在 OpenAPI 上先於 llm-generate；仍含 tab/unit/quiz/llm-grade / grade-result
app.include_router(zip_router)
# 評分相關 API
app.include_router(grade_router)
# Exam 測驗 API
app.include_router(exam_router)
# 個人分析 API
app.include_router(person_analysis_router)
# 課程分析 API
app.include_router(course_analysis_router)
# 使用者 API
app.include_router(users_router)
# 系統設定 API
app.include_router(system_settings_router)
# Log 查詢 API
app.include_router(log_router)


# 定義根路徑 / 的 GET 端點，用於健康檢查
@app.get("/")
def read_root(_person_id: PersonId):
    # 回傳 JSON 物件，表示伺服器運行中
    return {"status": "Server is running"}
