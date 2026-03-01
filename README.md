# AI Quiz Backend

FastAPI 後端：使用者登入（Supabase）、ZIP RAG、出題、評分。

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

OpenAI API key 由前端在請求中傳入，不需寫入 `.env`。

## Render 部署

在 Render 的 Service → **Environment** 新增與 `.env` 相同的變數：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

前端網址 `https://aiquizfrontend.vercel.app` 已加入 CORS 允許清單。
