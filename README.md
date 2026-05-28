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

評分請求（`POST /rag/tab/unit/quiz/llm-grade`、`POST /exam/tab/quiz/llm-grade` 等）請求 body 中學生作答欄位為 **`quiz_answer`**（仍相容舊欄位名 `answer`）；寫入 **`Rag_Quiz`／`Exam_Quiz` 的 `answer_content`**，批改評語寫入 **`answer_critique`**。已無獨立的 `Rag_Answer`／`Exam_Answer` 表，亦無 `quiz_grade`／`answer_grade` 欄位。

## Render 部署

在 Render 的 Service → **Environment** 新增與 `.env` 相同的變數：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

預設前端網址 `https://myquiz-ai.vercel.app` 已加入 CORS 允許清單（若再改網域請同步更新 `main.py` 或設定 `CORS_EXTRA_ORIGINS`）。

---

## API 回傳格式

所有端點皆需 query 參數 `person_id`（部分端點另需 `course_id`）。

### 通用錯誤格式

HTTP 4xx / 5xx 時統一回傳：

```json
{ "detail": "錯誤說明文字" }
```

評分非同步端點（llm-grade）在排程前檢查到錯誤時，直接回傳相應 HTTP 狀態碼加上：

```json
{ "error": "錯誤說明文字" }
```

---

### API 目錄

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | [`/`](#get-) | 健康檢查 |
| **使用者** | | |
| GET | [`/user/users`](#get-userusers) | 列出所有使用者 |
| POST | [`/user/users`](#post-userusers) | 新增使用者 |
| POST | [`/user/users/batch`](#post-userusersbatch) | 批次新增使用者 |
| PUT | [`/user/users/delete`](#put-userusersdelete) | 軟刪除使用者 |
| POST | [`/user/login`](#post-userlogin) | 登入 |
| PATCH | [`/user/profile`](#patch-userprofile) | 更新個人資料 |
| **RAG ZIP 管理** | | |
| GET | [`/rag/tabs`](#get-ragtabs) | 列出 Rag（含 units→quizzes） |
| POST | [`/rag/tab/create`](#post-ragtabcreate) | 建立 Rag |
| POST | [`/rag/tab/create-upload-zip`](#post-ragtabcreate-upload-zip) | 建立 Rag 並上傳 ZIP |
| PUT | [`/rag/tab/tab-name`](#put-ragtabtab-name) | 更新 Rag tab_name |
| PUT | [`/rag/tab/delete/{rag_tab_id}`](#put-ragtabdeleterag_tab_id) | 軟刪除 Rag |
| POST | [`/rag/tab/upload-zip`](#post-ragtabupload-zip) | 上傳 ZIP |
| POST | [`/rag/tab/build-rag-zip`](#post-ragtabbuild-rag-zip) | 打包 RAG ZIP（NDJSON 串流） |
| GET | [`/rag/tab/units`](#get-ragtabunits) | 列出 Rag_Unit（含 quizzes） |
| PUT | [`/rag/tab/unit/unit-name`](#put-ragtabunitunit-name) | 更新 Rag_Unit unit_name |
| GET | [`/rag/tab/unit/mp3-file`](#get-ragtabunitmp3-file) | 取得音訊（unit_type=3） |
| GET | [`/rag/tab/unit/youtube-url`](#get-ragtabunityoutube-url) | 取得 YouTube URL（unit_type=4） |
| POST | [`/rag/tab/unit/quiz/create`](#post-ragtabunitquizcreate) | 新增空白 Rag_Quiz |
| PUT | [`/rag/tab/unit/quiz/quiz-name`](#put-ragtabunitquizquiz-name) | 更新 Rag_Quiz quiz_name |
| PUT | [`/rag/tab/quiz/delete/{rag_quiz_id}`](#put-ragtabquizdeleterag_quiz_id) | 軟刪除 Rag_Quiz |
| **RAG 出題與評分** | | |
| POST | [`/rag/tab/unit/quiz/llm-generate`](#post-ragtabunitquizllm-generate) | LLM 出題 |
| POST | [`/rag/tab/unit/quiz/llm-generate-db`](#post-ragtabunitquizllm-generate) | LLM 出題（沿用 DB prompt） |
| POST | [`/rag/tab/unit/quiz/llm-generate-followup`](#post-ragtabunitquizllm-generate) | LLM 追問出題 |
| POST | [`/rag/tab/unit/quiz/llm-generate-followup-db`](#post-ragtabunitquizllm-generate) | LLM 追問出題（沿用 DB prompt） |
| POST | [`/rag/tab/unit/quiz/llm-grade`](#post-ragtabunitquizllm-grade) | 非同步評分（202 + job_id） |
| POST | [`/rag/tab/unit/quiz/llm-grade-db`](#post-ragtabunitquizllm-grade) | 非同步評分（沿用 DB prompt） |
| GET | [`/rag/tab/unit/quiz/grade-result/{job_id}`](#get-ragtabunitquizgrade-resultjob_id) | 輪詢評分結果 |
| POST | [`/rag/tab/unit/quiz/for-exam`](#post-ragtabunitquizfor-exam) | 更新 for_exam 標記 |
| GET | [`/rag/unit/text`](#get-ragunittext) | 取得文字單元逐字稿 |
| GET | [`/rag/unit/mp3-file`](#get-ragunitmp3-file) | 取得音訊與同資料夾文字檔逐字稿 |
| GET | [`/rag/unit/youtube-url`](#get-raguniityoutube-url) | 解析 YouTube URL 與第二行起逐字稿 |
| **測驗** | | |
| GET | [`/exam/tabs`](#get-examtabs) | 列出 Exam（含 quizzes） |
| GET | [`/exam/rag-for-exams`](#get-examrag-for-exams) | 列出 for_exam RAG 單元 |
| POST | [`/exam/tab/create`](#post-examtabcreate) | 建立 Exam |
| PUT | [`/exam/tab/tab-name`](#put-examtabtab-name) | 更新 Exam tab_name |
| PUT | [`/exam/tab/delete/{exam_tab_id}`](#put-examtabdeleteexam_tab_id) | 軟刪除 Exam |
| POST | [`/exam/tab/quiz/create`](#post-examtabquizcreate) | 新增空白 Exam_Quiz |
| POST | [`/exam/tab/quiz/llm-generate`](#post-examtabquizllm-generate) | LLM 出題 |
| POST | [`/exam/tab/quiz/llm-generate-followup`](#post-examtabquizllm-generate) | LLM 追問出題 |
| POST | [`/exam/tab/quiz/create-llm-generate`](#post-examtabquizllm-generate) | 建立並 LLM 出題 |
| POST | [`/exam/tab/quiz/create-llm-generate-followup`](#post-examtabquizllm-generate) | 建立並 LLM 追問出題 |
| POST | [`/exam/tab/quiz/llm-grade`](#post-examtabquizllm-grade) | 非同步評分（202 + job_id） |
| GET | [`/exam/tab/quiz/grade-result/{job_id}`](#get-examtabquizgrade-resultjob_id) | 輪詢評分結果 |
| POST | [`/exam/tab/quiz/rate`](#post-examtabquizrate) | 更新 quiz_rate |
| **課程分析** | | |
| GET | [`/course-analysis/quizzes`](#get-course-analysisquizzes) | 列出全部 Exam_Quiz |
| **個人分析** | | |
| GET | [`/person-analysis/quizzes/{person_id}`](#get-person-analysisquizzesperson_id) | 個人作答分析（含弱點報告） |
| **Log** | | |
| GET | [`/log/logs`](#get-loglogs) | 列出 API 呼叫紀錄 |
| **系統設定** | | |
| GET | [`/system-settings/person_analysis_user_prompt_text`](#get-system-settingsperson_analysis_user_prompt_text) | 取得個人分析 prompt |
| PUT | [`/system-settings/person_analysis_user_prompt_text`](#put-system-settingsperson_analysis_user_prompt_text) | 寫入個人分析 prompt |

---

### 健康檢查

#### `GET /`

確認伺服器正常運行（需帶 query `person_id`）。

```json
{ "status": "Server is running" }
```

---

### 使用者 `/user`

#### `GET /user/users`

列出所有未刪除使用者。

```json
{
  "users": [
    {
      "user_id": 1,
      "person_id": "string",
      "name": "string",
      "user_type": 3,
      "llm_api_key": "string | null",
      "user_metadata": null,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "count": 1
}
```

---

#### `POST /user/users`

新增單一使用者。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "llm_api_key": null,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

#### `POST /user/users/batch`

批次新增使用者（user_type 固定 3，密碼預設 0000）。

```json
{
  "created": [
    {
      "user_id": 1,
      "person_id": "string",
      "name": "string",
      "user_type": 3,
      "llm_api_key": null,
      "user_metadata": null,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "failed": [
    { "person_id": "string", "detail": "person_id 已存在" }
  ],
  "created_count": 1,
  "failed_count": 0
}
```

---

#### `PUT /user/users/delete`

軟刪除指定使用者（courses 固定為空列表）。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "llm_api_key": null,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

#### `POST /user/login`

登入成功時回傳使用者資訊與所有課程列表。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "llm_api_key": "string | null",
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": [
    {
      "course_user_id": 1,
      "course_id": 1,
      "course_name": "string",
      "user_type": 3
    }
  ]
}
```

---

#### `PATCH /user/profile`

更新個人資料（name、user_type、llm_api_key）；courses 固定為空列表。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "llm_api_key": "string | null",
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

### RAG ZIP 管理 `/rag`

#### `GET /rag/tabs`

列出 Rag（含 units→quizzes）。音訊單元含 `mp3_audio_url`；YouTube 單元含 `youtube_url_api`。

```json
{
  "rags": [
    {
      "rag_id": 1,
      "rag_tab_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "course_id": 1,
      "local": false,
      "deleted": false,
      "file_metadata": { "filename": "...", "second_folders": [], "file_size": 1.23 },
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "units": [
        {
          "rag_unit_id": 1,
          "rag_tab_id": "string",
          "person_id": "string",
          "course_id": 1,
          "unit_name": "string",
          "folder_combination": "string",
          "unit_type": 1,
          "repack_file_name": "string",
          "rag_file_name": "string",
          "rag_file_size": 1.23,
          "rag_chunk_size": 1000,
          "rag_chunk_overlap": 200,
          "transcription": "string",
          "text_file_name": "string",
          "mp3_file_name": "string",
          "youtube_url": "string",
          "deleted": false,
          "updated_at": "2024-01-01T00:00:00+08:00",
          "created_at": "2024-01-01T00:00:00+08:00",
          "mp3_audio_url": "/rag/tab/unit/mp3-file?rag_tab_id=...&rag_unit_id=...&course_id=...",
          "youtube_url_api": "/rag/tab/unit/youtube-url?rag_tab_id=...&rag_unit_id=...&course_id=...",
          "quizzes": [
            {
              "rag_quiz_id": 1,
              "rag_tab_id": "string",
              "rag_unit_id": 1,
              "person_id": "string",
              "quiz_name": "string",
              "quiz_user_prompt_text": "string",
              "quiz_content": "string",
              "quiz_hint": "string",
              "quiz_answer_reference": "string",
              "answer_user_prompt_text": "string",
              "answer_content": "string",
              "answer_critique": "string | null",
              "for_exam": false,
              "follow_up": false,
              "deleted": false,
              "updated_at": "2024-01-01T00:00:00+08:00",
              "created_at": "2024-01-01T00:00:00+08:00"
            }
          ]
        }
      ]
    }
  ],
  "count": 1
}
```

> `mp3_audio_url` 僅 unit_type=3 且 mp3_file_name 非空時出現；`youtube_url_api` 僅 unit_type=4 且 youtube_url 非空時出現。

---

#### `POST /rag/tab/create`

建立一筆 Rag。

```json
{
  "rag_id": 1,
  "rag_tab_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /rag/tab/create-upload-zip`

建立 Rag 並上傳 ZIP（multipart/form-data）。

```json
{
  "rag_id": 1,
  "rag_tab_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "created_at": "2024-01-01T00:00:00+08:00",
  "file_metadata": {
    "rag_id": 1,
    "rag_tab_id": "string",
    "created_at": "2024-01-01T00:00:00+08:00",
    "filename": "upload.zip",
    "second_folders": ["folder1", "folder2"],
    "file_size": 1.23
  }
}
```

---

#### `PUT /rag/tab/tab-name`

更新 Rag 的 tab_name。

```json
{
  "rag_id": 1,
  "rag_tab_id": "string",
  "person_id": "string",
  "tab_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/delete/{rag_tab_id}`

軟刪除 Rag 及其 Rag_Unit，並刪除 Storage 資料夾。

```json
{
  "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
  "rag_tab_id": "string",
  "person_id": "string",
  "rag_updated": true,
  "folder_deleted": true
}
```

---

#### `POST /rag/tab/upload-zip`

上傳 ZIP 並更新 Rag.file_metadata（multipart/form-data）。

```json
{
  "rag_id": 1,
  "rag_tab_id": "string",
  "created_at": "2024-01-01T00:00:00+08:00",
  "filename": "upload.zip",
  "second_folders": ["folder1", "folder2"],
  "file_size": 1.23
}
```

---

#### `POST /rag/tab/build-rag-zip`

NDJSON 串流回應（`application/x-ndjson`）。請以 `fetch` 逐行讀取，**勿**使用 `response.json()`。HTTP 狀態碼恒為 200，以最後一行 `type==="complete"` 的 `success` 判斷成敗。

**第 1 行 — start**
```json
{
  "type": "start",
  "total": 2,
  "source_rag_tab_id": "string",
  "unit_list": "folder1+folder2",
  "user_type": 1,
  "build_faiss_request": null,
  "repack_only": false,
  "allow_faiss": true
}
```

**第 N 行 — building（每單元前一行）**
```json
{
  "type": "building",
  "index": 1,
  "total": 2,
  "completed_before": 0,
  "filename": "folder1.zip"
}
```

**第 N+1 行 — unit（每單元結果）**
```json
{
  "type": "unit",
  "index": 1,
  "total": 2,
  "output": {
    "filename": "folder1.zip",
    "folder_combination": "folder1",
    "unit_name": "folder1",
    "repack_filename": "abc123.zip",
    "rag_filename": "abc123_rag.zip",
    "unit_type": 1,
    "rag_mode": "faiss",
    "transcript_plain": "string",
    "text_file_name": "string",
    "mp3_file_name": "string",
    "youtube_url": "string",
    "rag_chunk_size": 1000,
    "rag_chunk_overlap": 200,
    "file_size": 0.45,
    "rag_error": "string（僅失敗時出現）"
  }
}
```

> `rag_mode`：`"faiss"`（向量庫）、`"transcript_md"`（逐字稿 md ZIP）、`"repack_copy"`（與 repack 同內容）。
> `rag_chunk_size`／`rag_chunk_overlap` 於 unit_type≠1 時回傳 0。

**最後一行 — complete**
```json
{
  "type": "complete",
  "success": true,
  "source_rag_tab_id": "string",
  "unit_list": "folder1+folder2",
  "outputs": [ /* 同 unit.output */ ],
  "total": 2,
  "built_ok": 2,
  "built_failed": 0,
  "message": "RAG ZIP 建立失敗（請修正後重試）（僅失敗時出現）"
}
```

---

#### `GET /rag/tab/units`

依 rag_tab_id 列出所有 Rag_Unit（含 quizzes）。

```json
{
  "units": [
    {
      "rag_unit_id": 1,
      "rag_tab_id": "string",
      "person_id": "string",
      "unit_name": "string",
      "folder_combination": "string",
      "unit_type": 1,
      "repack_file_name": "string",
      "rag_file_name": "string",
      "rag_file_size": 1.23,
      "rag_chunk_size": 1000,
      "rag_chunk_overlap": 200,
      "transcription": "string",
      "text_file_name": "string",
      "mp3_file_name": "string",
      "youtube_url": "string",
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [ /* Rag_Quiz[] */ ]
    }
  ],
  "count": 1
}
```

---

#### `PUT /rag/tab/unit/unit-name`

更新 Rag_Unit 的 unit_name。

```json
{
  "rag_unit_id": 1,
  "rag_tab_id": "string",
  "person_id": "string",
  "unit_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `GET /rag/tab/unit/mp3-file`

依 rag_tab_id、rag_unit_id 回傳音訊（unit_type=3）。不需 person_id。

```json
{
  "rag_unit_id": 1,
  "rag_tab_id": "string",
  "audio_base64": "base64 encoded audio string",
  "media_type": "audio/mpeg",
  "filename": "audio.mp3",
  "transcription": "string"
}
```

---

#### `GET /rag/tab/unit/youtube-url`

依 rag_tab_id、rag_unit_id 回傳 YouTube URL（unit_type=4）。不需 person_id。

```json
{
  "rag_unit_id": 1,
  "rag_tab_id": "string",
  "youtube_url": "https://www.youtube.com/watch?v=...",
  "transcription": "string"
}
```

---

#### `POST /rag/tab/unit/quiz/create`

新增空白 Rag_Quiz（不呼叫 LLM）。

```json
{
  "rag_quiz_id": 1,
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "string",
  "quiz_user_prompt_text": "",
  "quiz_content": "",
  "quiz_hint": "",
  "quiz_answer_reference": "",
  "answer_user_prompt_text": "",
  "quiz_answer": "",
  "answer_content": "",
  "answer_critique": null,
  "for_exam": false,
  "follow_up": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/unit/quiz/quiz-name`

更新 Rag_Quiz 的 quiz_name。

```json
{
  "rag_quiz_id": 1,
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/quiz/delete/{rag_quiz_id}`

軟刪除 Rag_Quiz。

```json
{
  "message": "已將 Rag_Quiz 標記為刪除",
  "rag_quiz_id": 1,
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "rag_quiz_updated": true,
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

### RAG 出題與評分 `/rag`

#### `POST /rag/tab/unit/quiz/llm-generate`
#### `POST /rag/tab/unit/quiz/llm-generate-db`
#### `POST /rag/tab/unit/quiz/llm-generate-followup`
#### `POST /rag/tab/unit/quiz/llm-generate-followup-db`

LLM 出題後更新 Rag_Quiz 並回傳出題結果。`follow_up` 在 followup 端點為 true。

```json
{
  "rag_quiz_id": 1,
  "quiz_name": "string",
  "quiz_content": "題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "transcription": "逐字稿（unit_type=1 時為空字串）",
  "rag_output": {
    "rag_tab_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "follow_up": false
}
```

---

#### `POST /rag/tab/unit/quiz/llm-grade`
#### `POST /rag/tab/unit/quiz/llm-grade-db`

非同步評分。回傳 **HTTP 202**。

```json
{ "job_id": "uuid-string" }
```

---

#### `GET /rag/tab/unit/quiz/grade-result/{job_id}`

輪詢評分結果。`status` 為 `"pending"` | `"ready"` | `"error"`。

**pending / error 時**
```json
{
  "status": "pending",
  "result": null,
  "error": null
}
```

**ready 時**（另附 rag_quiz 整列）
```json
{
  "status": "ready",
  "result": {
    "quiz_comments": ["評語 Markdown 段落 1", "評語 Markdown 段落 2"],
    "rag_quiz_id": 1,
    "rag_answer_id": 1
  },
  "error": null,
  "rag_quiz": {
    "rag_quiz_id": 1,
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "person_id": "string",
    "quiz_name": "string",
    "quiz_content": "string",
    "quiz_hint": "string",
    "quiz_answer_reference": "string",
    "answer_content": "學生作答",
    "answer_critique": "批改評語純文字（非 JSON 物件）",
    "for_exam": false,
    "follow_up": false,
    "deleted": false,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  }
}
```

> `result.quiz_comments`：字串陣列（Markdown）。DB 的 `Rag_Quiz.answer_critique` 寫入合併後純文字，非 JSON 物件。`rag_answer_id` 為 `rag_quiz_id` 的向下相容別名。
```

---

#### `POST /rag/tab/unit/quiz/for-exam`

更新 Rag_Quiz.for_exam 標記。

```json
{
  "rag_quiz_id": 1,
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "string",
  "quiz_user_prompt_text": "string",
  "quiz_content": "string",
  "quiz_hint": "string",
  "quiz_answer_reference": "string",
  "answer_user_prompt_text": "string",
  "answer_content": "string",
  "quiz_answer": "string",
  "answer_critique": "string | null",
  "for_exam": true,
  "follow_up": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `GET /rag/unit/text`

回傳文字單元（unit_type=2）的 `text_file_name` 與 `transcription`（全文）。`folder_name` 與 `rag_unit_id` **二擇一**。

- **folder_name**：自 upload ZIP 讀取（與 build-rag-zip unit_type=2 一致）；須傳 `person_id`。
- **rag_unit_id**：自 DB 讀取，不需 `person_id`；DB 無逐字稿時改讀 upload ZIP。

```json
{
  "rag_tab_id": "string",
  "folder_name": "string",
  "rag_unit_id": 1,
  "text_file_name": "content.md",
  "transcription": "全文 Markdown 內容"
}
```

---

#### `GET /rag/unit/mp3-file`

自 upload ZIP 依 `folder_name` 回傳音訊與同資料夾內**至多一個**文字檔全文（與 build-rag-zip unit_type=3 一致）。需 person_id。

```json
{
  "rag_tab_id": "string",
  "folder_name": "string",
  "audio_base64": "base64 encoded audio string",
  "media_type": "audio/mpeg",
  "filename": "audio.mp3",
  "text_file_name": "transcript.md",
  "transcription": "文字檔全文"
}
```

---

#### `GET /rag/unit/youtube-url`

自 upload ZIP 解析 YouTube URL；文字檔**第一行**為 URL，**第二行起**為 `transcription`（與 build-rag-zip unit_type=4 一致）。需 person_id。

```json
{
  "rag_tab_id": "string",
  "folder_name": "string",
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "text_file_name": "unit.md",
  "transcription": "第二行起的逐字稿"
}
```

---

### 測驗 `/exam`

#### `GET /exam/tabs`

列出 Exam（含 quizzes）。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_tab_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "course_id": 1,
      "local": false,
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [
        {
          "exam_quiz_id": 1,
          "exam_tab_id": "string",
          "rag_tab_id": "string",
          "rag_unit_id": 1,
          "rag_quiz_id": 1,
          "person_id": "string",
          "course_id": 1,
          "unit_name": "string",
          "quiz_name": "string",
          "quiz_user_prompt_text": "string",
          "quiz_content": "string",
          "quiz_hint": "string",
          "quiz_answer_reference": "string",
          "quiz_rate": 0,
          "answer_user_prompt_text": "string",
          "answer_content": "string | null",
          "answer_critique": "string | null",
          "follow_up": false,
          "follow_up_exam_quiz_id": 0,
          "updated_at": "2024-01-01T00:00:00+08:00",
          "created_at": "2024-01-01T00:00:00+08:00",
          "follow_up_quiz": { /* 下一筆 follow_up Exam_Quiz，結構相同，可遞迴 */ }
        }
      ]
    }
  ],
  "count": 1
}
```

---

#### `GET /exam/rag-for-exams`

列出 for_exam=true 的 RAG 單元與題目。

```json
{
  "units": [
    {
      "rag_unit_id": 1,
      "rag_tab_id": "string",
      "unit_name": "string",
      "unit_type": 1,
      "for_exam": true,
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [
        {
          "rag_quiz_id": 1,
          "rag_tab_id": "string",
          "rag_unit_id": 1,
          "person_id": "string",
          "course_id": 1,
          "follow_up": false,
          "quiz_name": "string",
          "quiz_user_prompt_text": "string",
          "quiz_content": "string",
          "quiz_hint": "string",
          "quiz_answer_reference": "string",
          "answer_user_prompt_text": "string"
        }
      ]
    }
  ],
  "count": 1
}
```

---

#### `POST /exam/tab/create`

建立一筆 Exam。

```json
{
  "exam_id": 1,
  "exam_tab_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /exam/tab/tab-name`

更新 Exam 的 tab_name。

```json
{
  "exam_id": 1,
  "exam_tab_id": "string",
  "tab_name": "新名稱",
  "person_id": "string",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /exam/tab/delete/{exam_tab_id}`

軟刪除 Exam。

```json
{
  "message": "已將 Exam 標記為刪除",
  "exam_tab_id": "string",
  "person_id": "string"
}
```

---

#### `POST /exam/tab/quiz/create`

新增空白 Exam_Quiz（不呼叫 LLM）。回傳 Exam_Quiz 完整列。

```json
{
  "exam_quiz_id": 1,
  "exam_tab_id": "string",
  "rag_tab_id": "",
  "rag_unit_id": null,
  "rag_quiz_id": null,
  "person_id": "string",
  "course_id": 1,
  "follow_up": false,
  "follow_up_exam_quiz_id": 0,
  "unit_name": "",
  "quiz_name": "",
  "quiz_user_prompt_text": null,
  "quiz_content": "",
  "quiz_hint": "",
  "quiz_answer_reference": "",
  "quiz_rate": 0,
  "answer_user_prompt_text": null,
  "answer_content": null,
  "answer_critique": null,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /exam/tab/quiz/llm-generate`
#### `POST /exam/tab/quiz/create-llm-generate`

LLM 出題後更新 Exam_Quiz 並回傳出題結果。

```json
{
  "exam_quiz_id": 1,
  "quiz_name": "string",
  "quiz_content": "題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "unit_name": "string",
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "rag_quiz_id": 1,
  "transcription": "逐字稿（unit_type=1 時為空字串）",
  "rag_output": {
    "rag_tab_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /exam/tab/quiz/llm-generate-followup`
#### `POST /exam/tab/quiz/create-llm-generate-followup`

接續追問出題。`follow_up_exam_quiz_id > 0` 時回傳帶 follow_up 欄位的結果。

```json
{
  "exam_quiz_id": 2,
  "quiz_name": "string",
  "quiz_content": "追問題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "unit_name": "string",
  "rag_tab_id": "string",
  "rag_unit_id": 1,
  "rag_quiz_id": 1,
  "transcription": "string",
  "rag_output": {
    "rag_tab_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "follow_up": true,
  "follow_up_exam_quiz_id": 1,
  "quiz_history_list": [
    {
      "quiz_content": "string",
      "answer_content": "string",
      "quiz_answer_reference": "string",
      "answer_critique": "string"
    }
  ],
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

> `follow_up`、`follow_up_exam_quiz_id`、`quiz_history_list` 僅在 `follow_up_exam_quiz_id > 0` 且有傳入歷史問答時才出現。

---

#### `POST /exam/tab/quiz/llm-grade`

非同步評分。回傳 **HTTP 202**。

```json
{ "job_id": "uuid-string" }
```

---

#### `GET /exam/tab/quiz/grade-result/{job_id}`

輪詢評分結果。`status` 為 `"pending"` | `"ready"` | `"error"`。

**pending / error 時**
```json
{
  "status": "pending",
  "result": null,
  "error": null
}
```

**ready 時**（另附 exam_quiz 整列）
```json
{
  "status": "ready",
  "result": {
    "quiz_comments": ["評語 Markdown 段落 1", "評語 Markdown 段落 2"],
    "exam_quiz_id": 1
  },
  "error": null,
  "exam_quiz": {
    "exam_quiz_id": 1,
    "exam_tab_id": "string",
    "rag_tab_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "person_id": "string",
    "course_id": 1,
    "unit_name": "string",
    "quiz_name": "string",
    "quiz_content": "string",
    "quiz_hint": "string",
    "quiz_answer_reference": "string",
    "quiz_rate": 0,
    "answer_content": "學生作答",
    "answer_critique": "批改評語純文字（非 JSON 物件）",
    "follow_up": false,
    "follow_up_exam_quiz_id": 0,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  }
}
```

> `result.quiz_comments`：字串陣列（Markdown）。DB 的 `Exam_Quiz.answer_critique` 寫入合併後純文字，非 JSON 物件。
```

---

#### `POST /exam/tab/quiz/rate`

更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。

```json
{
  "exam_quiz_id": 1,
  "quiz_rate": 0,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00",
  "message": "已更新 quiz_rate"
}
```

---

### 課程分析 `/course-analysis`

#### `GET /course-analysis/quizzes`

回傳全部 Exam_Quiz（依 exam_tab_id 分群），`weakness_report` 固定為 null。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_tab_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "quizzes": [ /* Exam_Quiz[]，結構同 GET /exam/tabs */ ]
    }
  ],
  "count": 1,
  "weakness_report": null
}
```

---

### 個人分析 `/person-analysis`

#### `GET /person-analysis/quizzes/{person_id}`

依 person_id 取得已作答 Exam_Quiz，並產生弱點報告。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_tab_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "quizzes": [ /* Exam_Quiz[]（含 answer_content 非空者），結構同 GET /exam/tabs */ ]
    }
  ],
  "count": 1,
  "weakness_report": "Markdown 弱點報告全文（LLM 成功時非 null）"
}
```

> `weakness_report` 為 LLM `message.content` 原文 Markdown；無 API Key、無可分析題目或失敗時為 null。

---

### Log `/log`

#### `GET /log/logs`

依 course_id 列出 API 呼叫紀錄，依 log_id 降冪。

```json
{
  "logs": [
    {
      "log_id": 1,
      "person_id": "string",
      "course_id": 1,
      "api": "POST /rag/tab/unit/quiz/llm-generate",
      "api_metadata": { "key": "value" },
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "count": 1
}
```

---

### 系統設定 `/system-settings`

#### `GET /system-settings/person_analysis_user_prompt_text`

取得個人分析 user prompt（所有有效使用者皆可讀取）。

```json
{
  "system_setting_id": 1,
  "person_analysis_user_prompt_text": "string（無設定時為 null）"
}
```

---

#### `PUT /system-settings/person_analysis_user_prompt_text`

寫入個人分析 user prompt（僅 user_type 1／2 可操作）。

```json
{
  "system_setting_id": 1,
  "person_analysis_user_prompt_text": "string"
}
```
