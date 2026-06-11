# Quiz 分析 API 文件

給前端串接用。本文件涵蓋兩支基於 **Quiz 模組（Quiz_QA）** 的分析 API：

| API | 路由前綴 | 分析範圍 | 說明 |
|-----|----------|----------|------|
| **User Analysis** | `/v1/user-analyses` | 個人 × 課程 | 單一學生在某課程的 Quiz_QA 作答弱點分析 |
| **Quiz Analysis** | `/v1/quiz-analyses` | 試卷（全體學生） | 單份試卷（`quiz_page_id`）所有學生的作答彙整分析 |

> **與 Person / Course Analysis 的差異**：`person-analyses` / `course-analyses` 分析 RAG 出題（Exam_Quiz）；
> 本文件兩支 API 分析 **Bank 出題試卷（Quiz_QA）**，金鑰獨立（`quiz-api-key`）、模型獨立（`quiz-llm-model`）。

---

## 0. 共通約定

- **Base URL**：所有端點都在 `/v1` 之下
- **驗證**：每支 API 都要帶
  ```
  Authorization: Bearer <access_token>
  ```
- **軟刪**：刪除都是 `deleted=true`，不是真的刪列
- **LLM 設定**：分析使用 Quiz 模組的金鑰與模型，需先設定
  - `PUT /v1/quiz/llm-api-key`　`{ "api_key": "sk-..." }`
  - `PUT /v1/quiz/llm-model`　`{ "llm_model": "gpt-5.4" }`
- **分析指令（可選）**：教師可在 Course_Setting 預設 LLM 分析指令
  - `user_analysis_user_prompt_text`：個人弱點分析指令
  - `quiz_analysis_user_prompt_text`：測驗作答分析指令

### 使用模式（兩支 API 皆相同）

```
1. POST /{prefix}                     → 建一筆空白分析列，取得 id
2. POST /{prefix}/{id}/llm-analysis   → 呼叫 LLM 產生報告，寫入該列
3. GET  /{prefix}                     → 列出所有分析結果（含最新 Quiz_QA 彙整）
4. PATCH /{prefix}/{id}               → 改名
5. DELETE /{prefix}/{id}              → 軟刪
```

### `quizzes` 回傳結構（分組格式）

GET 列表與 llm-analysis 都會附上即時彙整的作答資料，格式如下：

```json
[
  {
    "quiz_page_id": "u123_260610153000",
    "tab_name": "第一回測驗",
    "group_count": 1,
    "groups": [
      {
        "quiz_group_id": 34,
        "group_name": "細胞題組",
        "unit_name": "第三章 細胞",
        "qa_count": 3,
        "qas": [
          {
            "quiz_qa_id": 101,
            "question_content": "題幹…",
            "question_answer_reference": "參考答案…",
            "answer_content": "學生作答…",
            "answer_critique": "批改評語…",
            "answer_rate": 1,
            "question_rate": 0,
            "...": "其餘 Quiz_QA 欄位"
          }
        ]
      }
    ]
  }
]
```

---

## 1. User Analysis（個人弱點分析）

分析特定學生在某課程中所有 Quiz_QA 的作答表現，LLM 根據作答與批改內容產生個人弱點報告。

**`person_id`** = 呼叫者（即被分析的學生）登入帳號

---

### 1.1 列出分析結果　`GET /v1/user-analyses`

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者登入帳號 |
| `course_id` | ✅ | 課程 ID |

**Response 200**

```json
{
  "person_id": "u123",
  "course_id": 5,
  "analyses": [
    {
      "user_analysis_id": 1,
      "person_id": "u123",
      "course_id": 5,
      "analysis_name": "第一次分析",
      "analysis_user_prompt_text": "請分析學生弱點…（當下的教師指令）",
      "analysis_prompt_text": "請分析學生弱點…（同上，DB 欄位名）",
      "analysis_text": "## 弱點分析\n根據作答…（Markdown 報告）",
      "quizzes": [ { "...": "分組格式，見上方說明" } ],
      "created_at": "2026-06-11T20:00:00+08:00",
      "updated_at": "2026-06-11T20:05:00+08:00"
    }
  ],
  "count": 1
}
```

- `quizzes`：即時自 DB 彙整該學生在此課程的已作答 Quiz_QA（有 `answer_content`、`answer_critique` 或 `answer_rate ≠ 0` 者）
- `analysis_text` 為空字串時，表示此列剛建立、尚未產生報告

---

### 1.2 新增空白分析列　`POST /v1/user-analyses`

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者登入帳號 |
| `course_id` | ✅ | 課程 ID |
| `analysis_name` | — | 分析名稱（未填存空字串） |

**Response 201**

```json
{
  "message": "已新增 User_Analysis 列",
  "user_analysis_id": 1,
  "person_id": "u123",
  "course_id": 5,
  "analysis_name": "第一次分析",
  "created_at": "2026-06-11T20:00:00+08:00",
  "updated_at": "2026-06-11T20:00:00+08:00"
}
```

拿到 `user_analysis_id` 後呼叫 1.3 產生報告。

---

### 1.3 產生弱點報告（呼叫 LLM）　`POST /v1/user-analyses/{user_analysis_id}/llm-analysis`

**Path**：`user_analysis_id`（POST 1.2 取得）

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者登入帳號（須與該列 owner 相符） |
| `course_id` | ✅ | 課程 ID（須與該列相符） |

**Response 200**

```json
{
  "user_analysis_id": 1,
  "quizzes": [ { "...": "已作答 Quiz_QA 分組，同 1.1 的 quizzes 格式" } ],
  "count": 2,
  "weakness_report": "## 個人弱點分析\n\n根據您在「第一回測驗」…（Markdown）",
  "llm_error": null,
  "analysis_llm_model": "gpt-5.4"
}
```

| 欄位 | 說明 |
|------|------|
| `weakness_report` | LLM 產生的弱點報告 Markdown；`null` 表示無法產生（見 `llm_error`） |
| `llm_error` | 錯誤訊息；`null` 表示成功 |
| `count` | 試卷數（`quiz_page_id` 個數） |

**常見 `llm_error` 情境**

| 情境 | 訊息 |
|------|------|
| 無已作答題目 | `"無已作答或已評級 Quiz_QA，無法產生弱點報告…"` |
| 未設定 API Key | `"未設定 API Key：PUT /v1/quiz/llm-api-key…"` |
| 無權限（403） | HTTP 403 `"無權寫入該 User_Analysis 列"` |

> 報告產生後會自動寫回 `User_Analysis` 列的 `analysis_text`，之後 GET 1.1 可直接讀取，**不需要再呼叫 LLM**。

---

### 1.4 更名　`PATCH /v1/user-analyses/{user_analysis_id}`

**Query**：`person_id`（必）

**Body**
```json
{ "analysis_name": "更新後的名稱" }
```

**Response 200**
```json
{
  "message": "已更新 User_Analysis 分析名稱",
  "user_analysis_id": 1,
  "person_id": "u123",
  "course_id": 5,
  "analysis_name": "更新後的名稱",
  "updated_at": "2026-06-11T20:10:00+08:00"
}
```

---

### 1.5 軟刪除　`DELETE /v1/user-analyses/{user_analysis_id}`

**Query**：`person_id`（必）

**Response 200**
```json
{
  "message": "已將 User_Analysis 標記為刪除",
  "user_analysis_id": 1,
  "person_id": "u123",
  "course_id": 5,
  "updated_at": "2026-06-11T20:15:00+08:00"
}
```

---

## 2. Quiz Analysis（測驗作答分析）

分析單份試卷（`quiz_page_id`）中**全體學生**的 Quiz_QA 作答，LLM 根據所有學生的答題狀況產生整體分析報告（哪題最難、常見錯誤等）。

**`person_id`** = 建立分析的人（通常是教師），**不是**學生

---

### 2.1 列出分析結果　`GET /v1/quiz-analyses`

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者（教師）登入帳號 |
| `quiz_page_id` | ✅ | 試卷識別碼 |

**Response 200**

```json
{
  "quiz_page_id": "u123_260610153000",
  "analyses": [
    {
      "quiz_analysis_id": 1,
      "person_id": "u123",
      "quiz_page_id": "u123_260610153000",
      "course_id": 5,
      "analysis_name": "期中考分析",
      "analysis_user_prompt_text": "請分析哪些題目學生普遍答錯…",
      "analysis_prompt_text": "請分析哪些題目學生普遍答錯…",
      "analysis_text": "## 測驗作答分析\n\n共收到 15 位學生作答…（Markdown）",
      "quizzes": [ { "...": "所有學生已作答 Quiz_QA 分組，見說明" } ],
      "created_at": "2026-06-11T21:00:00+08:00",
      "updated_at": "2026-06-11T21:05:00+08:00"
    }
  ],
  "count": 1
}
```

- `quizzes`：即時彙整該試卷所有學生的已作答 Quiz_QA（依 `quiz_group_id` 分組）
- 同一道題（`question_content`）會出現多筆（對應不同學生），可用 `person_id` 欄位區分

---

### 2.2 新增空白分析列　`POST /v1/quiz-analyses`

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者（教師）登入帳號 |
| `quiz_page_id` | ✅ | 試卷識別碼 |
| `course_id` | ✅ | 課程 ID（供後續 llm-analysis 取 API Key） |
| `analysis_name` | — | 分析名稱（未填存空字串） |

**Response 201**

```json
{
  "message": "已新增 Quiz_Analysis 列",
  "quiz_analysis_id": 1,
  "person_id": "u123",
  "quiz_page_id": "u123_260610153000",
  "course_id": 5,
  "analysis_name": "期中考分析",
  "created_at": "2026-06-11T21:00:00+08:00",
  "updated_at": "2026-06-11T21:00:00+08:00"
}
```

---

### 2.3 產生分析報告（呼叫 LLM）　`POST /v1/quiz-analyses/{quiz_analysis_id}/llm-analysis`

**Path**：`quiz_analysis_id`（POST 2.2 取得）

**Query**

| 參數 | 必填 | 說明 |
|------|------|------|
| `person_id` | ✅ | 呼叫者登入帳號（須與該列 owner 相符） |
| `quiz_page_id` | ✅ | 試卷識別碼（須與該列相符） |
| `course_id` | ✅ | 課程 ID（取 `quiz-api-key` / `quiz-llm-model`） |

**Response 200**

```json
{
  "quiz_analysis_id": 1,
  "quiz_page_id": "u123_260610153000",
  "quizzes": [ { "...": "所有學生已作答 Quiz_QA 分組" } ],
  "count": 1,
  "weakness_report": "## 測驗作答分析\n\n根據 15 位學生作答，第 2 題答錯率最高…（Markdown）",
  "llm_error": null,
  "analysis_llm_model": "gpt-5.4"
}
```

| 欄位 | 說明 |
|------|------|
| `weakness_report` | LLM 產生的分析報告 Markdown；`null` 表示無法產生 |
| `llm_error` | 錯誤訊息；`null` 表示成功 |
| `count` | 題組數 |

**常見 `llm_error` 情境**

| 情境 | 訊息 |
|------|------|
| 無學生作答 | `"該試卷尚無學生已作答或已評級 Quiz_QA，無法產生分析報告…"` |
| 未設定 API Key | `"未設定 API Key：PUT /v1/quiz/llm-api-key…"` |
| `quiz_page_id` 不符（403） | HTTP 403 `"無權寫入該 Quiz_Analysis 列"` |

---

### 2.4 更名　`PATCH /v1/quiz-analyses/{quiz_analysis_id}`

**Query**：`person_id`（必）

**Body**
```json
{ "analysis_name": "期末考分析" }
```

**Response 200**
```json
{
  "message": "已更新 Quiz_Analysis 分析名稱",
  "quiz_analysis_id": 1,
  "person_id": "u123",
  "quiz_page_id": "u123_260610153000",
  "course_id": 5,
  "analysis_name": "期末考分析",
  "updated_at": "2026-06-11T21:10:00+08:00"
}
```

---

### 2.5 軟刪除　`DELETE /v1/quiz-analyses/{quiz_analysis_id}`

**Query**：`person_id`（必）

**Response 200**
```json
{
  "message": "已將 Quiz_Analysis 標記為刪除",
  "quiz_analysis_id": 1,
  "person_id": "u123",
  "quiz_page_id": "u123_260610153000",
  "course_id": 5,
  "updated_at": "2026-06-11T21:15:00+08:00"
}
```

---

## 3. 典型前端流程

### User Analysis（學生視角）

```
（前提）學生已完成 Quiz 作答與批改（Quiz_QA.answer_content / answer_critique）

1. POST /v1/user-analyses?person_id=u123&course_id=5
        body（可不帶）→ 取得 user_analysis_id

2. POST /v1/user-analyses/{user_analysis_id}/llm-analysis
        ?person_id=u123&course_id=5
        → 取得 weakness_report（Markdown）

3. GET  /v1/user-analyses?person_id=u123&course_id=5
        → 列出所有歷次分析結果（含 quizzes 彙整）

（改名）PATCH /v1/user-analyses/{user_analysis_id}?person_id=u123
（刪除）DELETE /v1/user-analyses/{user_analysis_id}?person_id=u123
```

### Quiz Analysis（教師視角）

```
（前提）學生已完成該試卷的 Quiz 作答，教師想彙整全班表現

1. POST /v1/quiz-analyses
        ?person_id=teacher01&quiz_page_id=u123_260610153000&course_id=5
        → 取得 quiz_analysis_id

2. POST /v1/quiz-analyses/{quiz_analysis_id}/llm-analysis
        ?person_id=teacher01&quiz_page_id=u123_260610153000&course_id=5
        → 取得 weakness_report（Markdown，彙整所有學生答題狀況）

3. GET  /v1/quiz-analyses
        ?person_id=teacher01&quiz_page_id=u123_260610153000
        → 列出所有歷次分析結果

（改名）PATCH /v1/quiz-analyses/{quiz_analysis_id}?person_id=teacher01
（刪除）DELETE /v1/quiz-analyses/{quiz_analysis_id}?person_id=teacher01
```

---

## 4. 與其他分析 API 對照

| | person-analyses | course-analyses | **user-analyses** | **quiz-analyses** |
|--|--|--|--|--|
| 資料來源 | Exam_Quiz (RAG) | Exam_Quiz (RAG) | **Quiz_QA (Bank)** | **Quiz_QA (Bank)** |
| 主要 scope | person_id + course_id | course_id | person_id + course_id | **quiz_page_id**（單份試卷） |
| 分析對象 | 個人 (RAG 題) | 整門課 (RAG 題) | 個人 (Bank 試卷題) | **全體學生**（某份試卷） |
| API Key 設定路徑 | `PUT /exam/llm-api-key` | `PUT /rag/llm-api-key` | `PUT /quiz/llm-api-key` | `PUT /quiz/llm-api-key` |
| 分析指令 Course_Setting key | `person_analysis_user_prompt_text` | `course_analysis_user_prompt_text` | `user_analysis_user_prompt_text` | `quiz_analysis_user_prompt_text` |

---

## 5. 資料庫新增表格（需在 Supabase 執行）

> 詳細 DDL 見 `docs/QUIZ_SCHEMA.sql` 末段，在 Supabase SQL Editor 執行。

**`User_Analysis`**：儲存個人弱點分析結果

| 欄位 | 型別 | 說明 |
|------|------|------|
| `user_analysis_id` | bigint PK | 主鍵 |
| `person_id` | varchar(255) | 被分析學生 |
| `course_id` | bigint | 課程 |
| `analysis_name` | varchar(255) | 分析名稱 |
| `analysis_prompt_text` | text | 產生報告當下的教師指令快照 |
| `analysis_text` | text | LLM 弱點報告 Markdown |
| `deleted` | boolean | 軟刪標記 |

**`Quiz_Analysis`**：儲存測驗作答分析結果

| 欄位 | 型別 | 說明 |
|------|------|------|
| `quiz_analysis_id` | bigint PK | 主鍵 |
| `person_id` | varchar(255) | 建立者（教師） |
| `quiz_page_id` | varchar(255) | 所分析的試卷識別碼 |
| `course_id` | bigint | 課程 |
| `analysis_name` | varchar(255) | 分析名稱 |
| `analysis_prompt_text` | text | 產生報告當下的教師指令快照 |
| `analysis_text` | text | LLM 分析報告 Markdown |
| `deleted` | boolean | 軟刪標記 |
