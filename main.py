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
# 引入 CORS 中介軟體，用於處理跨域請求
from fastapi.middleware.cors import CORSMiddleware

# 引入各子模組的路由器，並以別名區分（zip_router 等）
from routers.zip import router as zip_router
# 評分相關路由（含 generate-quiz、quiz-grade、quiz-grade-result）
from routers.grade import router as grade_router
# Exam 測驗相關路由（exams、create-exam、generate-quiz、quiz-grade 等）
from routers.exam import router as exam_router
# 個人分析路由（依 person_id 查詢測驗與弱點報告）
from routers.person_analysis import router as person_analysis_router
# 課程分析路由（Exam_Quiz 全部內容）
from routers.course_analysis import router as course_analysis_router
# 使用者相關路由（登入、個人資料、使用者列表）
from routers.users import router as users_router
# 系統設定路由（LLM API Key 的 GET/PUT）
from routers.system_settings import router as system_settings_router

# 建立 FastAPI 應用程式實例
app = FastAPI()

# 註冊 CORS 中介軟體，允許前端跨域呼叫 API，避免瀏覽器 CORS 或 "Failed to fetch" 錯誤
# 若出現 502，回應來自 Render 代理（逾時約 30 秒），不會帶 CORS 標頭
# 評分已改為非同步：POST /rag/quiz-grade 回傳 202 + job_id，請用 GET /rag/quiz-grade-result/{job_id} 輪詢結果
app.add_middleware(
    # 使用 CORSMiddleware 中介軟體
    CORSMiddleware,
    # 允許的來源網域列表，前端可由此發送請求
    allow_origins=[
        "http://localhost:8080",           # 本地開發（localhost）
        "http://127.0.0.1:8080",           # 本地開發（127.0.0.1）
        "https://kevin7261.github.io",     # GitHub Pages 前端
        "https://aiquizfrontend.vercel.app",  # Vercel 前端
    ],
    # 允許攜帶 cookie、認證等憑證
    allow_credentials=True,
    # 允許所有 HTTP 方法（GET、POST、PUT、PATCH、DELETE 等）
    allow_methods=["*"],
    # 允許所有 HTTP 標頭
    allow_headers=["*"],
)

# 依序掛載各路由至 app；順序影響 Swagger API 文件顯示
# zip_router 先掛載，使 generate-quiz / quiz-grade / quiz-grade-result 出現在 rag 群組最下面
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


# 定義根路徑 / 的 GET 端點，用於健康檢查
@app.get("/")
def read_root():  # 處理根路徑請求
    # 回傳 JSON 物件，表示伺服器運行中
    return {"status": "Server is running"}
