# MyQuiz.ai-backend

FastAPI 後端（MyQuiz.ai）：使用者登入（Supabase）、ZIP RAG、出題、評分。

## 本機設定

1. 複製環境變數範例並填入實際值：
   ```bash
   cp .env.example .env
   ```
2. 編輯 `.env`，必填：
   - `SUPABASE_URL`：Supabase 專案網址（`https://xxxxx.supabase.co`）
   - `SUPABASE_ANON_KEY` 或 `SUPABASE_SERVICE_ROLE_KEY`（到 [Supabase Dashboard](https://supabase.com/dashboard) → Project Settings → API 取得）
3. 安裝依賴並啟動：
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

LLM／Deepgram API key 請在 `.env`（本機）或部署平台 **Environment** 設定 `LLM_API_KEY` 或 `OPENAI_API_KEY`、`DEEPGRAM_API_KEY`。Rag 相關出題／評分可另於使用者資料 `User.llm_api_key` 填寫；未填時 fallback 至上述環境變數。

評分請求（`POST /rag/tab/unit/quiz/llm-grade`、`POST /exam/tab/quiz/grade` 等）學生作答欄位為 **`quiz_answer`**（仍相容舊欄位名 `answer`）；寫入資料庫 **`Rag_Answer`／`Exam_Answer` 的 `quiz_answer` 欄**。若 `Exam_Answer` 尚未有該欄，請在 Supabase 將作答欄更名或新增為 `quiz_answer`。

## Render 部署

在 Render 的 Service → **Environment** 新增與 `.env` 相同的變數：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

預設前端網址 `https://myquiz-ai.vercel.app` 已加入 CORS 允許清單（若再改網域請同步更新 `main.py` 或設定 `CORS_EXTRA_ORIGINS`）。
