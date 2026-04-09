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

LLM API key 由使用者於系統設定（`GET/PUT /system-settings/llm-api-key`，表僅一筆、不需 person_id）填寫，不需在 API 請求中傳入，也不需寫入 `.env`。

評分請求（`POST /rag/tab/quiz/grade`、`POST /exam/tab/quiz/grade` 等）學生作答欄位為 **`quiz_answer`**（仍相容舊欄位名 `answer`）；寫入資料庫 **`Rag_Answer`／`Exam_Answer` 的 `quiz_answer` 欄**。若 `Exam_Answer` 尚未有該欄，請在 Supabase 將作答欄更名或新增為 `quiz_answer`。

## Render 部署

在 Render 的 Service → **Environment** 新增與 `.env` 相同的變數：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

預設前端網址 `https://myquiz-ai.vercel.app` 已加入 CORS 允許清單（若再改網域請同步更新 `main.py` 或設定 `CORS_EXTRA_ORIGINS`）。
